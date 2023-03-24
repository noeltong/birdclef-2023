from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import random
from utils import audio


class BirdCLEFDataset(Dataset):
    def __init__(self, data_lst, mode, sample_rate=32000, resample_rate=16000, num_channels=3, duration=8000):
        super().__init__()
        self.data = data_lst
        self.sample_rate = sample_rate
        self.resample_rate = resample_rate
        self.channel = num_channels
        self.duration = duration
        self.mode = mode

    def __getitem__(self, idx):
        path, label = self.data[idx]
        aud = audio.read_file(path)

        aud = audio.resample(aud, self.resample_rate)
        aud = audio.rechannel(aud, self.channel)
        aud = audio.pad_trunc(aud, self.resample_rate, self.duration)

        if self.mode == 'train':
            sg1 = self.sg_augment(aud)
            sg2 = self.sg_augment(aud)

            return sg1, sg2
        
        elif self.mode == 'tune':
            sg = self.sg_augment(aud)

            return sg, label
    
    def __len__(self):
        return len(self.data)
    
    def sg_augment(self, aud):
        aud = audio.time_shift(aud, self.shift_factor)
        sg = audio.spectrogram(aud, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=None)
        sg = audio.spectro_augment(sg, max_mask_pct=0.1, n_freq_masks=self.n_masks, n_time_masks=self.n_masks)
    

def get_dataloader(config, mode):

    csv_path = config.data.csv_path

    data = []

    with open(csv_path, 'r') as f:
        for line in f.readlines():
            label, path = line.split(',')
            label = int(label.strip())
            path = path.strip()

            data.append([path, label])

    random.shuffle(data)

    idx = int(0.9 * len(data))
    train_data, test_data = data[:idx], data[idx:]

    train_dataset = BirdCLEFDataset(train_data, mode)
    test_dataset = BirdCLEFDataset(test_data, mode)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        sampler=test_sampler,
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler, test_sampler