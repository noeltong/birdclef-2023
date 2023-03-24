from torch import nn
from torchaudio import transforms as T
import torch
from utils import audio


class BirdDataAugment(nn.Module):
    def __init__(self, resample_rate=16000, num_channels=3, duration=8000, shift_factor=0.25, n_mels=256, n_fft=1024, n_masks=2) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.channel = num_channels
        self.duration = duration
        self.shift_factor = shift_factor
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_masks = n_masks

    def forward(self, aud: torch.Tensor) -> torch.Tensor:
        aud = audio.resample(aud, self.resample_rate)
        aud = audio.rechannel(aud, self.channel)
        aud = audio.pad_trunc(aud, self.duration)
        aud = audio.time_shift(aud, self.shift_factor)
        sg = audio.spectrogram(aud, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=None)
        sg = audio.spectro_augment(sg, max_mask_pct=0.1, n_freq_masks=self.n_masks, n_time_masks=self.n_masks)

        return sg