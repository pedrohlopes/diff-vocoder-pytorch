from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler


class Trainer:
    
    def __init__(self, model, configs):
        self.configs = configs
        self.model = model
        self.writer = SummaryWriter('{}/run_{}'.format(self.configs['expdir'],timestamp))
        
    def _train_one_step(self, batch):
        
        loss = self.model(batch)
        loss.backward()
        return loss
        
    
    def _train_one_epoch(self, epoch_index, tb_writer, optimizer, training_loader, print_interval):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs= data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            loss = self.train_one_step(inputs)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % print_interval == 0:
                last_loss = running_loss / print_interval # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def _evaluate_one_step():
        print('to be implemented')
        
    def __evaluate_one_epoch():
        print('to be implemented')

    
    def train(self, optimizer, training_loader): # to do: add validation loader and logic
        self.model.train(True)
        for epoch in range(self.configs['total_epochs']):
            print('EPOCH {}:'.format(epoch+ 1))
            avg_loss = self._train_one_epoch(epoch, self.writer,optimizer,training_loader,self.configs['print_interval']) # to do: add validation loader and logic
            print('LOSS train: {}'.format(avg_loss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('TrainingLoss',
                            { 'Training' : avg_loss})
            self.writer.flush()