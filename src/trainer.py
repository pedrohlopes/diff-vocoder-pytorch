from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from audio_diffusion_pytorch import DiffusionUpsampler, UNetV0, VDiffusion, VSampler
import torch


class Trainer:
    
    def __init__(self, model, configs):
        self.configs = configs
        self.model = model
        self.writer = SummaryWriter('{}/run'.format(self.configs['expdir']))
        
    def _train_one_step(self, batch):
        
        loss = self.model(batch)
        loss.backward()
        return loss
        
    
    def _train_one_epoch(self, epoch_index, tb_writer, optimizer, training_loader, configs):
        running_loss = 0.
        last_loss = 0.
        print_interval = configs['print_interval']
        save_interval = configs['save_interval']
        checkpoint_path = configs['checkpoint_path']
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs= data

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            loss = self._train_one_step(inputs)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % print_interval == 0:
                last_loss = running_loss / print_interval # loss per batch
                print('  batch {}/{} loss: {}'.format(i,len(training_loader), last_loss))
                tb_x = epoch_index * len(training_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            if (epoch_index*len(training_loader) + i) % save_interval == 0:
                print('Saving model...')
                torch.save(self.model.state_dict(), checkpoint_path + ('model_%d.pt' % (epoch_index*len(training_loader) + i)))
                

        return last_loss
    
    def _evaluate_one_step():
        print('to be implemented')
        
    def __evaluate_one_epoch():
        print('to be implemented')

    
    def train(self, optimizer, training_loader): # to do: add validation loader and logic
        self.model.train(True)
        for epoch in range(self.configs['total_epochs']):
            print('EPOCH {}:'.format(epoch))
            avg_loss = self._train_one_epoch(epoch, self.writer,optimizer,training_loader,self.configs) # to do: add validation loader and logic
            print('LOSS train: {}'.format(avg_loss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('TrainingLoss',
                            { 'Training' : avg_loss})
            self.writer.flush()