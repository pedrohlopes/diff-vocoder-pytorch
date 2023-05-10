import wave
import csv
import os

def get_wav_info(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(frame_rate)
    return channels, sample_width, frame_rate, n_frames, duration

def construct_csv(wav_dir, output_csv, duration, hop):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time', 'duration', 'filename'])
        for wav_file in os.listdir(wav_dir):
            if wav_file.endswith('.wav'):
                wav_path = os.path.join(wav_dir, wav_file)
                channels, sample_width, frame_rate, n_frames, wav_duration = get_wav_info(wav_path)
                start_time = 0
                while start_time + duration <= wav_duration:
                    writer.writerow([start_time, duration, wav_file])
                    start_time += hop

if __name__ == '__main__':

    construct_csv('/path/to/wav/files', 'output.csv', 2.0, 0.01)
