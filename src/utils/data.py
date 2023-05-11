import wave
import torch


def get_wav_info(wav_file):
    with wave.open(wav_file, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(frame_rate)
    return channels, sample_width, frame_rate, n_frames, duration


def collate_fn(batch, output_length):
    batch_size = len(batch)
    out = torch.zeros(batch_size, 1, output_length)
    for sample_num, sample in enumerate(batch):
        out[sample_num, :, : sample.shape[-1]] = sample
    return out
