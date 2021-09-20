import os
import numpy as np
import pandas as pd
from pandas._config import config
from tqdm.notebook import tqdm
from joblib import delayed, Parallel
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .audio import AudioProcessor
from .text import TextProcessor

class TTSDataset(Dataset):
    def __init__(self, json_path, audio_processor, text_processor, div_steps=5, use_phonemes=False, use_precomputed_mels=False, mel_path=None):
        super().__init__()
        self.json_path = json_path
        self.div_steps = div_steps
        self.use_phonemes = use_phonemes
        self.use_precomputed_mels = use_precomputed_mels

        self.meta = pd.read_json(json_path, lines=True, orient='records')
        self.meta = self.meta.sort_values('duration', ascending=False).reset_index(drop=True)
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        if use_precomputed_mels and 'mel_path' not in self.meta.columns:
            assert mel_path is not None, "Please specify mel_path."
            os.makedirs(mel_path, exist_ok=True)
            self.mel_path = mel_path
            self.process_audios()

    def process_audios(self):
        def _save_feat(file):
            mel = self.audio_processor(file)
            np.save(self.mel_path + file.split('/')[-1][:-4], mel)

        files = self.meta.path
        _ = Parallel(n_jobs=20)(delayed(_save_feat)(file) for file in tqdm(files))
        self.meta['mel_path'] = self.meta['path'].apply(
            lambda x: self.mel_path + x.split('/')[-1][:-4] + '.npy'
        )
        self.meta.to_json(self.json_path, lines=True, orient='records')

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        samp = self.meta.loc[idx]
        if self.use_precomputed_mels and 'mel_path' in self.meta.columns:
            mel = np.load(samp['mel_path'])
        else:
            mel = self.audio_processor(samp['path'])
        mel = np.pad(mel, ((0, 0), (1, 1)))
        phones = self.text_processor(samp['sentence'], to_phones=self.use_phonemes, to_indices=True)
        return phones, mel.T

    def fn(self, batch):
        phones = [b[0] for b in batch]
        len_phones = torch.tensor([len(p) for p in phones])
        M_phones = len_phones.max().int()
        phones = torch.Tensor([np.pad(p, (0, M_phones - len(p)), constant_values=self.text_processor.pad_id) for p in phones]).long()

        mels = [b[1] for b in batch]
        len_mels = torch.tensor([len(m) for m in mels])
        M_mels = len_mels.max().int()
        if M_mels % self.div_steps != 0:
            M_mels += self.div_steps- (M_mels % self.div_steps)
        mels = torch.Tensor([np.pad(m, [(0, M_mels - len(m)), (0,0)]) for m in mels])

        sorted_idx = torch.argsort(len_phones, descending=True)
        return phones[sorted_idx], len_phones[sorted_idx], mels[sorted_idx], len_mels[sorted_idx]

def get_dataloader(dataset, batch_size, n_jobs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.fn) 
