from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler
from trainer import Trainer
from dataset import AudioDataset
from torch import optim
from torch.utils.data import DataLoader
import yaml
import torch
from glob import glob

LEN = 2 ** 18

def collate_fn(batch):
    bsz = len(batch)
    out = torch.zeros(bsz, 1, LEN)
    for i, x in enumerate(batch):
        out[i, :, :x.shape[1]] = x # torch.from_numpy(x)
    return out.to('cuda:0')

if __name__ == '__main__':
    
    
    model = DiffusionUpsampler(
        net_t=UNetV0, # The model type used for diffusion
        upsample_factor=4, # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
        in_channels=1, # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
    )
    model.to('cuda:0')
    
    with open('../configs.yml', 'r') as stream:
        try:
            configs=yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as e:
            print('invalid config file:',e)
            
    optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'])
    training_filelist = glob(configs['wavs_path'] + '*.wav')    
    training_set = AudioDataset(duration= 5.0,filelist=training_filelist)
    training_loader = DataLoader(training_set, batch_size=configs['batch_size'], shuffle=True,collate_fn=collate_fn)
    trainer = Trainer(model,configs)
    trainer.train(optimizer,training_loader)