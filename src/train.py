from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler
from models.trainer import Trainer
from models.dataset import AudioDataset
from utils.data import collate_fn
from torch.utils.data import DataLoader
import yaml
import torch
from glob import glob
import os


if __name__ == "__main__":
    with open("../configs.yml", "r") as configs:
        params = yaml.safe_load(configs)
        print(params)
    # todo colocar canais e outros parametros do modelo no YAML
    model = DiffusionUpsampler(
        net_t=UNetV0,  # The model type used for diffusion
        upsample_factor=params[
            "resample_factor"
        ],  # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
        in_channels=1,  # U-Net: number of input/output (audio) channels
        channels=[
            8,
            32,
            64,
            128,
            256,
            512,
            512,
            1024,
            1024,
        ],  # U-Net: channels at each layer
        factors=[
            1,
            4,
            4,
            4,
            2,
            2,
            2,
            2,
            2,
        ],  # U-Net: downsampling and upsampling factors at each layer
        items=[
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            4,
            4,
        ],  # U-Net: number of repeating items at each layer
        diffusion_t=VDiffusion,  # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
    )
    params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    files = glob(os.path.join(params["wavs_path"], "*.wav"))
    dataset = AudioDataset(snippet_duration=params["snippet_duration"], files=files)
    train, validation = dataset.make_splits(params)
    trainer = Trainer(model, optimizer, params)
    trainer.train(train, validation)
