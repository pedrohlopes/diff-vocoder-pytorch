import torch
from torch.utils.data import Dataset
import librosa 
import pandas as pd

class AudioDataset(Dataset):

    def __init__(self,basepath,metadata_path):
        self.audio_dir = basepath
        self.annotation = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self,idx):
        start_time,duration, filename = self.annotation.iloc[idx].values()
        
        audio,_ = librosa.load(filename,sr=None,offset=start_time,duration=duration)
        return torch.Tensor(audio)

