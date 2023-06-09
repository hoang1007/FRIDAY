from warnings import warn
from typing import List, Tuple, Any
import os
import torch
import torchaudio
torchaudio.set_audio_backend('soundfile')
import torch_audiomentations as T
from torch.utils.data import Dataset, DataLoader


class WakeWordDataset(Dataset):
    def __init__(
        self, data: List[Tuple[str, int]], sample_rate: int = 16000, transform=None
    ):
        self._data = data
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        path, label = self._data[idx]
        waveform, sample_rate = torchaudio.load(path)
        # waveform = waveform.float()
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


class WakeWordDataConstructor:
    def __init__(
        self,
        root: str = "data",
        pos_prefix: str = "pos",
        neg_prefix: str = "neg",
        noise_prefix: str = "noise",
        sample_rate: int = 16000,
        batch_size: int = 8,
        train_ratio: float = 0.75,
    ):
        self.root = root
        self.pos_prefix = pos_prefix
        self.neg_prefix = neg_prefix
        self.noise_prefix = noise_prefix

        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.train_data, self.val_data = self._prepare_data()

    def _get_audio_paths(self, path: str):
        paths = []
        for filename in os.listdir(path):
            if filename.endswith((".wav", ".mp3")):
                paths.append(os.path.join(path, filename))

        return paths

    def random_sampling(self, data: List[Any], train_ratio: float = 0.75):
        indices = torch.randperm(len(data))
        train_size = int(len(data) * train_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        return train_data, val_data

    def _prepare_data(self):
        pos_paths = self._get_audio_paths(os.path.join(self.root, self.pos_prefix))
        neg_paths = self._get_audio_paths(os.path.join(self.root, self.neg_prefix))
        noise_paths = self._get_audio_paths(os.path.join(self.root, self.noise_prefix))

        # Use noise as negative samples
        neg_paths.extend(noise_paths)

        pos_data = [(path, 1) for path in pos_paths]
        neg_data = [(path, 0) for path in neg_paths]

        pos_train, pos_val = self.random_sampling(pos_data, self.train_ratio)
        neg_train, neg_val = self.random_sampling(neg_data, self.train_ratio)

        print("Train/val: %d/%d positive, %d/%d negative" % (len(pos_train), len(pos_val), len(neg_train), len(neg_val)))

        # If data is skewed, balance it
        if len(pos_train) / len(neg_train) < 0.5:
            print("Data is skewed (train/val %.2f), balancing..." % (len(pos_train) / len(neg_train)))
            while len(pos_train) / len(neg_train) < 0.5:
                pos_train.extend(pos_train)
            print("Train/val: %d/%d positive, %d/%d negative" % (len(pos_train), len(pos_val), len(neg_train), len(neg_val)))

        train_data = pos_train + neg_train
        val_data = pos_val + neg_val

        return train_data, val_data
    
    def collate_fn(self, batch):
        waveforms, labels = zip(*batch)
        return waveforms, torch.tensor(labels)

    def train_dataloader(self):
        noise_path = os.path.join(self.root, self.noise_prefix)
        transforms = T.Compose(
            [
                T.AddBackgroundNoise(noise_path),
                T.AddColoredNoise(),
                T.Gain(),
                T.Shift(),
            ],
            p=0.5,
        )

        def do_transform(waveform):
            waveform = waveform.unsqueeze_(0)
            waveform = transforms(waveform, sample_rate=self.sample_rate)
            return waveform.squeeze_(0)

        return DataLoader(
            WakeWordDataset(
                self.train_data,
                sample_rate=self.sample_rate,
                transform=do_transform,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            WakeWordDataset(self.val_data, sample_rate=self.sample_rate),
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
        )
