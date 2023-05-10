from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler
from audio_diffusion_pytorch.utils import downsample
import librosa
import torch
import soundfile as sf
upsampler = DiffusionUpsampler(
    net_t=UNetV0, # The model type used for diffusion
    upsample_factor=4, # The upsample factor (e.g. 16 can be used for 3kHz to 48kHz)
    in_channels=1, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)
LEN = 2 ** 14
def collate_fn(batch):
    bsz = len(batch)
    out = torch.zeros(bsz, 1, LEN)
    for i, x in enumerate(batch):
        out[i, :, :x.shape[1]] = x # torch.from_numpy(x)
    return out

upsampler.load_state_dict(torch.load('/home/pedro.lopes/diff_vocoder_pytorch/exp/checkpoints/model_1800.pt'))
filename = '/home/pedro.lopes/tts_data/voz_base_44kHz_16bit/wavs_48/JornalNacional_01_00041.wav'
start_time = 2
duration = 5.0
audio_orig,_ = librosa.load(filename,sr=None,offset=start_time,duration=duration)
print(audio_orig.shape)
audio_tensor = torch.unsqueeze(torch.unsqueeze(torch.Tensor(audio_orig),0),0)
downsampled_audio = downsample(audio_tensor,factor=4)
print(downsampled_audio.shape)
sf.write('downsampled.wav',downsampled_audio[0][0],samplerate=48000//4)
sample = upsampler.sample(collate_fn(downsampled_audio), num_steps=35)
sf.write('output.wav',sample[0][0],samplerate=48000)