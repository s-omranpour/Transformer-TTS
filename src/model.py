import torch
from torch import nn
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from pytorch_msssim import SSIM, MS_SSIM


from .modules import EncoderPrenet, DecoderPrenet, DecoderPostNet, RelativePositionalEncoding

class Encoder(nn.Module):
    def __init__(self, n_vocab, d_emb, d_hidden, n_head, d_inner, n_encoder_layers=3, n_prenet_layers=3, kernel=5, dropout=0.2):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_emb = RelativePositionalEncoding(d_hidden, max_len=20000)
        self.prenet = EncoderPrenet(n_vocab, d_emb, d_hidden, n_prenet_layers, kernel, dropout)
#         self.prenet = nn.Embedding(n_vocab, d_hidden)
        self.dropout = nn.Dropout(dropout)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_hidden, n_head, 
            dim_feedforward=d_inner, dropout=0.1, activation='gelu', batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_encoder_layers)

    def forward(self, x, length_mask=None):
        x = self.dropout(self.prenet(x) + self.pos_emb(x) * self.alpha)
        x = self.enc(x, src_key_padding_mask=length_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n_mel, d_hidden, n_head, d_inner, outputs_per_step, n_decoder_layers=3, n_postnet_layers=3, kernel=5, dropout=0.2):
        super().__init__()
        self.n_mel = n_mel
        self.outputs_per_step = outputs_per_step

        self.pos_emb = RelativePositionalEncoding(d_hidden, max_len=20000)
        self.alpha = nn.Parameter(torch.ones(1))
        self.prenet = DecoderPrenet(n_mel, d_hidden*2, d_hidden, dropout)
        self.proj = nn.Linear(d_hidden, d_hidden)

        dec_layer = nn.TransformerDecoderLayer(d_hidden, n_head, dim_feedforward=d_inner, dropout=0.1, activation='gelu', batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        self.mel_linear = nn.Linear(d_hidden, n_mel * outputs_per_step)
        self.stop_linear = nn.Linear(d_hidden, 2)

        self.postnet = DecoderPostNet(n_mel, d_hidden, kernel, n_postnet_layers, dropout)
#         self.postnet = nn.Linear(n_mel, n_mel)


    def forward(self, x, memory, att_mask=None, trg_length_mask=None, memory_length_mask=None):
        
        x = self.proj(self.prenet(x) + self.pos_emb(x) * self.alpha)
        x = self.dec(x, memory, 
                     tgt_mask=att_mask, tgt_key_padding_mask=trg_length_mask, memory_key_padding_mask=memory_length_mask)
        
        n_batch = x.shape[0]
        n_time = x.shape[1]
        mel_out = self.mel_linear(x).view(n_batch, n_time * self.outputs_per_step, self.n_mel)
        post_out = (self.postnet(mel_out.transpose(1, 2)) + mel_out.transpose(1, 2)).transpose(1, 2)
#         post_out = self.postnet(mel_out)
        stop_out = self.stop_linear(x)
        return mel_out, post_out, stop_out


class TransformerTTS(pl.LightningModule):
    def __init__(self, config, audio_processor, text_processor):
        super().__init__()
        self.config = config
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.l1 = nn.L1Loss(reduction='none')
#         self.l2 = nn.MSELoss(reduction='none')
#         self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True)
#         self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=1)
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=torch.Tensor([0.8, 0.2]))

        self.encoder = Encoder(**config['encoder'])
        self.decoder = Decoder(**config['decoder'])
            

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['max_epochs'], eta_min=0.)
        return [opt], [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, characters, mels=None, character_lengths=None, mel_lengths=None):
        trg_length_mask = self._generate_length_mask(mel_lengths)
        memory_length_mask = self._generate_length_mask(character_lengths)
        att_mask = None if mels is None else self._generate_causal_att_mask(mels.shape[1]).to(mels.device)
        memory = self.encoder(characters, memory_length_mask)
        return self.decoder(mels, memory, att_mask, trg_length_mask, memory_length_mask)

    def _generate_length_mask(self, lengths=None):
        mask = None
        if lengths is not None:
            mask = torch.ones(len(lengths), max(lengths)).bool().to(lengths.device)
            for i,l in enumerate(lengths):
                mask[i, :l] = False
        return mask

    def _generate_causal_att_mask(self, n):
        return torch.triu(torch.ones(n,n), diagonal=1).bool()

    def step(self, batch, batch_idx, mode='train'):
        characters, character_lengths, mels, mel_lengths = batch
        mel_out, post_out, stop_out = self.forward(characters, mels[:,:-1], character_lengths, mel_lengths-1)
        
        self.log('encoder alpha', self.encoder.alpha)
        self.log('decoder alpha', self.decoder.alpha)
        
        mask = 1-self._generate_length_mask(mel_lengths-1).to(mel_lengths.device).float()
        mask_2d = mask.unsqueeze(2).repeat(1,1,80)
        
        l1_mel_loss = (self.l1(mel_out, mels[:,1:]) * mask_2d).mean()
        l1_post_loss = (self.l1(post_out, mels[:, 1:]) * mask_2d).mean()
#         l2_post_loss = (self.l2(post_out, mels) * mask_2d).mean()
#         l2_mel_loss = (self.l2(mel_out, mels) * mask_2d).mean()
#         ssim_mel_loss = self.ssim(mel_out.unsqueeze(1), mels.unsqueeze(1))
#         ssim_post_loss = self.ssim(post_out.unsqueeze(1), mels.unsqueeze(1))
        stop_loss = (self.ce(stop_out.transpose(1,2), nn.functional.pad(mask, (0,1,0,0))[:,1:].long()) * mask).mean()

        loss = l1_mel_loss + l1_post_loss + stop_loss #+ ssim_mel_loss + ssim_post_loss + l2_post_loss + l2_mel_loss

        self.log(mode + '_l1_mel', l1_mel_loss.item())
        self.log(mode + '_l1_post', l1_post_loss.item())
#         self.log(mode + '_ssim_mel', ssim_mel_loss.item())
#         self.log(mode + '_ssim_post', ssim_post_loss.item())
#         self.log(mode + '_l2_mel', l2_mel_loss.item())
#         self.log(mode + '_l2_post', l2_post_loss.item())
        self.log(mode + '_stop_ce', stop_loss.item())
        self.log(mode + '_loss', loss.item())
        # if mode =='val':
        #     fig_spec = self.make_figure(linear_outputs[0])
        #     fig_attn = self.make_figure(attn[0])
        #     self.logger.experiment.add_figure('spectrogram-%d' % batch_idx, fig_spec, batch_idx)
        #     self.logger.experiment.add_figure('attn-%d' % batch_idx, fig_attn, batch_idx)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode='val')

    def make_figure(self, x):
        fig, ax = plt.subplots()
        im = ax.imshow(
            x.T.detach().cpu().numpy(),
            aspect='auto', origin='lower', interpolation='none'
        )
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def synthesize(self, prompt, max_len=500):
        self.eval()   
        prompt = self.text_processor(prompt, to_phones=False, to_indices=True)
        prompt = torch.tensor(prompt).unsqueeze(0).long()

        mels = torch.zeros([1,1, 80])
        with torch.no_grad():
            for i in tqdm(range(max_len)):
                mel_out, post_out, stop_out = self.forward(prompt, mels)
                mels = torch.cat([mels, mel_out[:,-1:,:]], dim=1)
                if torch.argmax(stop_out[:, -1]) == 0:
                    break
                
        return mels[0], post_out
