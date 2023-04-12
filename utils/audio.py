# Modified from https://www.kaggle.com/code/bibanh/lb-0-72-resnet34-melspectrogram-stage-1-training


import torchaudio
import torch
from torch import nn
import random
from torchaudio import transforms as T
import soundfile as sf


def read_file(path):
    # sig, sr = torchaudio.load(path)
    sig, sr = sf.read(path, always_2d=True, dtype='float32')
    sig = torch.from_numpy(sig)
    return (sig, sr)


def rechannel(aud, new_ch):
    sig, sr = aud

    if sig.shape[0] != new_ch:
        if new_ch == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig, sig])
    else:
        resig = sig

    return (resig, sr)


def resample(aud, new_sr):
    sig, sr = aud

    if sr != new_sr:
        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1,:])
        resig = resig.reshape(1, -1)
    else:
        resig = sig

    return (resig, new_sr)


def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if sig_len > max_len:
        max_start = sig_len - max_len
        start = random.randint(0, max_start - 1)
        end = start + max_len
        sig = sig[:, start : end]

    elif sig_len < max_len:
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)


def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)

    return (sig.roll(shift_amt), sr)


def spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)

    return spec


def spectro_augment(spec, max_mask_pct=0.25, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    # mask_value = spec.mean()

    time_mask_param = int(max_mask_pct * n_steps)
    freq_mask_param = int(max_mask_pct * n_mels)

    aug_fn = nn.Sequential()

    for _ in range(n_time_masks):
        aug_fn.append(T.TimeMasking(time_mask_param, iid_masks=True))
    for _ in range(n_freq_masks):
        aug_fn.append(T.FrequencyMasking(freq_mask_param, iid_masks=True))

    
    # time_mask = T.TimeMasking(time_mask_param, iid_masks=True)
    # freq_mask = T.FrequencyMasking(freq_mask_param, iid_masks=True)

    # for _ in range(n_time_masks):
    aug_spec = aug_fn(spec)

    return aug_spec
