from torch.utils.tensorboard import SummaryWriter
from audio_diffusion_pytorch.utils import downsample
from datetime import datetime
from tqdm import tqdm
import soundfile as sf
import torch
import os


class Trainer:
    def __init__(self, model, optimizer, training_params):
        self.params = training_params
        self.model = model
        self.optimizer = optimizer
        logs_path = os.path.join(
            training_params["expdir"], datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        os.mkdir(logs_path)
        tensorboard_path = os.path.join(logs_path, "run")
        self.writer = SummaryWriter(tensorboard_path)
        self.checkpoints_path = os.path.join(logs_path, "checkpoints")
        os.mkdir(self.checkpoints_path)
        self.samples_path = os.path.join(logs_path, "samples")
        os.mkdir(self.samples_path)
        self.loss_moving_average = 0.0

    def _train_step(self, batch):
        loss = self.model(batch)
        loss.backward()
        return loss

    def _update_loss_mv_average(self, loss, batch_number):
        if batch_number % self.params["log_interval"]:
            self.loss_moving_average = 0
        self.loss_moving_average += loss.item() / self.params["log_interval"]

    def _print_loss(self, board, index, value):
        self.writer.add_scalar(board, value, index)

    def _save_model(self, num_executed_batches):
        print("\nSaving model ...")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoints_path, f"model_{num_executed_batches}.pt"),
        )

    def _train_epoch(self, epoch_index, train_data):
        pbar = tqdm(enumerate(train_data))
        for batch_number, batch in pbar:
            batch = batch.to(self.params["device"])
            self.optimizer.zero_grad()
            loss = self._train_step(batch)
            self.optimizer.step()
            self._update_loss_mv_average(loss, batch_number)
            num_executed_batches = epoch_index * len(train_data) + batch_number
            if (batch_number + 1) % self.params["log_interval"] == 0:
                self._print_loss(
                    "Loss/train", num_executed_batches, self.loss_moving_average
                )
                pbar.set_description(
                    f"Batch {batch_number}/{epoch_index} --- loss {self.loss_moving_average}"
                )
            if (num_executed_batches + 1) % self.params["save_interval"] == 0:
                self._save_model(num_executed_batches)

    def validate(self, validation_data):
        validation_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in validation_data:
                batch = batch.to(self.params["device"])
                loss = self.model(batch)
                validation_loss += loss.item()
        return validation_loss / len(validation_data)

    def _print_sample(self, validation_audio, step):
        sample = self.model.sample(validation_audio, self.params["num_steps_sample"])
        sf.write(
            os.path.join(self.samples_path, f"sample_{step}.wav"),
            sample,
            self.params["sample_sr"],
        )
        self.writer.add_audio("Sample Audio", sample, step, self.params["sample_sr"])
        stft = torch.stft(sample, n_fft=2048, center=False)
        self.writer.add_audio("Audio STFT", stft, step)

    def train(self, train_data, validation_data):
        self.model.to(self.params["device"])
        for epoch in range(self.params["total_epochs"]):
            print(f"RUNNING EPOCH {epoch}/{self.params['total_epochs']}")
            self.model.train()
            self._train_epoch(epoch, train_data)
            validation_loss = self.validate(validation_data)
            print("Validation ")
            self._print_loss(
                "Loss/validation", epoch * len(train_data), validation_loss
            )
            downsampled_audio = downsample(
                next(iter(validation_data)), self.params["resample_factor"]
            )
            print(downsampled_audio.shape)
            self._print_sample(downsampled_audio, epoch * len(train_data))
