from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from utils.data import collate_fn
import torch
import numpy as np
import torchaudio
import math


class AudioDataset(Dataset):
    def __init__(self, snippet_duration, files):
        self.snippet_duration = snippet_duration
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        audio, fs = torchaudio.load(audio_file)
        snippet_len = int(self.snippet_duration * fs)
        if audio.shape[1] > snippet_len:
            # offset = np.random.randint(0, audio.shape[1] - snippet_len)
            offset = 2 * fs
        else:
            offset = 0
        return audio[:, offset : offset + snippet_len]

    def _get_snippet_len(self, params):
        min_len = params["snippet_duration"] * params["sample_sr"]
        return 2 ** (math.ceil(math.log(min_len, 2)))

    def make_splits(self, params, seed=1):
        # todo: desmarretar tamanhos dos splitsi
        train, validation = random_split(
            self, [1, 1], generator=torch.Generator().manual_seed(seed)
        )
        snippet_len = self._get_snippet_len(params)
        train_loader = DataLoader(
            dataset=train,
            shuffle=True,
            batch_size=params["batch_size"],
            collate_fn=lambda batch: collate_fn(batch, snippet_len),
        )
        validation_loader = DataLoader(
            dataset=validation,
            shuffle=True,
            batch_size=params["batch_size"],
            collate_fn=lambda batch: collate_fn(batch, snippet_len),
        )
        return train_loader, validation_loader
