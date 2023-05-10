from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torchaudio

class AudioDataset(Dataset):

    def __init__(self,duration,filelist):
        self.duration = duration
        self.filelist = filelist
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        audio_file = self.filelist[idx]
        audio, fs = torchaudio.load(audio_file)
        LEN = int(self.duration*fs)
        if audio.shape[1] > LEN:
            offset = np.random.randint(0, audio.shape[1] - LEN)
        else:
            offset = 0
        return audio[:, offset:offset + LEN]

