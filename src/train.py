from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler
from trainer import Trainer
from dataset import AudioDataset
from torch import optim
from torch.utils.data import DataLoader
import yaml


if __name__ == '__main__':
    
    
    model = DiffusionUpsampler(
        net_t=UNetV0, # The model type used for diffusion
        upsample_factor=3, # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
        in_channels=2, # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    with open('../configs.yml', 'r') as stream:
        try:
            configs=yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as e:
            print('invalid config file:',e)
            
        
    training_set = AudioDataset(configs['basepath'],configs['metadata_path'])
    training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
    trainer = Trainer(model)
    trainer.train(optimizer,training_loader)