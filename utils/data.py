from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import random
from utils import audio


class BirdCLEFDataset(Dataset):
    def __init__(self, data_lst):
        super().__init__()
        self.data = data_lst

    def __getitem__(self, idx):
        path, label = self.data[idx]
        aud = audio.read_file(path)

        return aud, label
    
    def __len__(self):
        return len(self.data)
    

def get_dataloader(config):

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

    train_dataset = BirdCLEFDataset(train_data)
    test_dataset = BirdCLEFDataset(test_data)

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