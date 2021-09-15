import torch
from torch import nn
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from .modules import EncoderPrenet, DecoderPrenet, DecoderPostNet, RelativePositionalEncoding

class Encoder(nn.Module):
    def __init__(self, n_vocab, d_emb, d_hidden, n_head, d_inner, n_encoder_layers=3, n_prenet_layers=3, kernel=5, dropout=0.2):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_emb = RelativePositionalEncoding(d_hidden, max_len=20000)
        self.prenet = EncoderPrenet(n_vocab, d_emb, d_hidden, n_prenet_layers, kernel, dropout)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_hidden, n_head, 
            dim_feedforward=d_inner, dropout=0.1, activation='relu', batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_encoder_layers)

    def forward(self, x, length_mask=None):
        x = self.prenet(x) + self.pos_emb(x) * self.alpha
        x = self.enc(x, src_key_padding_mask=length_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n_mel, d_hidden, n_head, d_inner, outputs_per_step, n_decoder_layers=3, n_postnet_layers=3, kernel=5, dropout=0.2):
        super().__init__()

        self.pos_emb = RelativePositionalEncoding(d_hidden, max_len=20000)
        self.alpha = nn.Parameter(torch.ones(1))
        self.prenet = DecoderPrenet(n_mel, d_hidden*2, d_hidden, dropout)
        self.proj = nn.Linear(d_hidden, d_hidden)

        dec_layer = nn.TransformerDecoderLayer(d_hidden, n_head, dim_feedforward=d_inner, dropout=0.1, activation='relu', batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        self.mel_linear = nn.Linear(d_hidden, n_mel * outputs_per_step)
        self.stop_linear = nn.Linear(d_hidden, 1)

        self.postnet = DecoderPostNet(n_mel, d_hidden, outputs_per_step, kernel, n_postnet_layers, dropout)


    def forward(self, x, memory, att_mask=None, trg_length_mask=None, memory_length_mask=None):
        x = self.prenet(x) + self.pos_emb(x) * self.alpha
        x = self.dec(x, memory, tgt_mask=att_mask, tgt_key_padding_mask=trg_length_mask, memory_key_padding_mask=memory_length_mask)
        mel_out = self.mel_linear(x)
        post_out = (self.postnet(mel_out.transpose(1, 2)) + mel_out.transpose(1, 2)).transpose(1, 2)
        stop_out = self.stop_linear(x)
        return post_out, mel_out, stop_out


class TransformerTTS(pl.LightningModule):
    def __init__(self, config, audio_processor, text_processor):
        super().__init__()
        self.config = config
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.encoder = Encoder(**config['encoder'])
        self.decoder = Decoder(**config['decoder'])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['max_epochs'], eta_min=0.)
        return [opt]#, [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, characters, mels=None, character_lengths=None, mel_lengths=None):
        trg_length_mask = self._generate_length_mask(mel_lengths)
        memory_length_mask = self._generate_length_mask(character_lengths)
        att_mask = self._generate_causal_att_mask(mels.shape[1])
        memory = self.encoder(characters, memory_length_mask)
        return self.decoder(mels, memory, att_mask, trg_length_mask, memory_length_mask)

    def _generate_length_mask(self, lengths=None):
        mask = None
        if lengths is not None:
            mask = torch.ones(len(lengths), max(lengths)).bool()
            for i,l in enumerate(lengths):
                mask[i, :l] = False
        return mask

    def _generate_causal_att_mask(self, n):
        return torch.triu(torch.ones(n,n), diagonal=1).bool()

    def step(self, batch, batch_idx, mode='train'):
        characters, character_lengths, mels, mel_lengths = batch
        post_out, mel_out, stop_out = self.forward(characters, mels, character_lengths, mel_lengths)
        
        l1_mel_loss = self.l1(mel_out, mels)   
        l2_mel_loss = self.l2(mel_out, mels)   
        l1_post_loss = self.l1(post_out, mels)
        l2_post_loss = self.l2(post_out, mels)

        loss = l1_mel_loss + l1_post_loss + l2_mel_loss + l2_post_loss
        self.log(mode + '_l1_mel', l1_mel_loss.item())
        self.log(mode + '_l1_post', l1_post_loss.item())
        self.log(mode + '_l2_mel', l2_mel_loss.item())
        self.log(mode + '_l2_post', l2_post_loss.item())
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
        prompt = torch.tensor(prompt + [self.text_processor.pad_id]*(5 - (len(prompt) % 5))).unsqueeze(0).long()

        mels = torch.zeros([1,1, 80])
        pbar = tqdm(range(max_len))
        with torch.no_grad():
            for i in pbar:
                post_out, mel_out, stopout = self.forward(prompt, mels)
                mels = torch.cat([mels, mel_out[:,-1:,:]], dim=1)
        return mels[0].T
