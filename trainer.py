import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import Model
from dataset import StimuliDataset

def load_args():
    parser = argparse.ArgumentParser(description='Leison study on AlexNet')
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs')    
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')    
    parser.add_argument('--init_lr', type=float,
                        help='Start learning rate')    
    parser.add_argument('--save_freq', type=int,
                        help='Save checkpoint every N epoch')
    parser.add_argument('--result_folder', type=str,
                        help='Where to save trained models.')
    parser.add_argument('--meta_path', type=str,
                        help="where meta.pkl is saved.")
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.load_data()

    def load_model(self):
        self.model = Model(pretrained=True)
        self.model.cuda()
        self.model.train()
        
    def load_data(self):
        dataset = StimuliDataset(self.args.meta_path, train=True)
        self.dataloader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            shuffle=True, num_workers=4)
        
    def init_training(self):
        self.optimizer = Adam(
            self.model.parameters(), lr=self.args.init_lr)
        self.criterion = nn.CrossEntropyLoss()        
        self.all_losses = []
        
    def train(self):
        self.init_training()
        self.save_checkpoint(0)
        for epoch in tqdm(range(1, self.args.num_epochs + 1)):
            for i, (ims, labels) in tqdm(enumerate(self.dataloader)):
                save_image(ims, f'train_{i}.png')
                self.optimizer.zero_grad()
                ims = ims.cuda()
                labels = labels.type(torch.long).cuda()
                preds = self.model(ims)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                self.all_losses.append(loss.item())
            if epoch % args.save_freq == 0:
                self.save_checkpoint(epoch)
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.all_losses
        }
        ckpt_path = os.path.join(            
            args.result_folder, f'{epoch}.pth')
        torch.save(checkpoint, ckpt_path)        
        

if __name__ == '__main__':
    args = load_args()
    trainer = Trainer(args)
    trainer.train()
