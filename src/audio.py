import numpy as np
import librosa

class AudioProcessor:
    def __init__(self, sr=22050, ref_level_db=35, min_level_db=-100, n_fft=1024, n_mels=80, hop_length=512, window='hann'):
        self.sr = sr
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.window = window

    def trim_silence(self, x):
        return librosa.effects.trim(x, top_db=self.ref_level_db, frame_length=self.n_fft, hop_length=self.hop_length)[0]

    def compute_mel(self, x):
        S = librosa.feature.melspectrogram(y=x, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)
        S = librosa.power_to_db(S, ref=np.max)
        return self.normalize(S)

    def compute_spec(self, x):
        S = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        S = librosa.amplitude_to_db(S) - self.ref_level_db
        return self.normalize(S)

    def normalize(self, x):
        return np.clip((x - self.min_level_db) / -self.min_level_db, 0, 1)

    def denormalize(self, x):
        return (np.clip(x, 0, 1) * -self.min_level_db) + self.min_level_db

    def spec_to_audio(self, spec):
        spec = self.denormalize(spec) + self.ref_level_db
        return librosa.griffinlim(spec, n_iter=50, hop_length=self.hop_length, win_length=self.n_fft, window=self.window)

    def mel_to_audio(self, mel, n_iter=50):
        mel = self.denormalize(mel)
        mel = librosa.db_to_power(mel)
        return librosa.feature.inverse.mel_to_audio(mel, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, n_iter=n_iter)
    
    def __call__(self, file):
        x, _ = librosa.load(file, sr=self.sr)
        x = self.trim_silence(x)
        return self.compute_mel(x)