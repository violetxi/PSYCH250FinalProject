import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader

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
    parser.add_argument('--data_root', type=str,
                        help="Where processed PT files are.")
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.load_data()

    def load_model(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)    # 4 classes in fLoc dataset        
        self.model.cuda()
        
    def load_data(self):
        dataset = StimuliDataset(self.args.data_root, train=True)
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
        for epoch in tqdm(range(self.args.num_epochs)):
            for i, (ims, labels) in tqdm(enumerate(self.dataloader)):
                self.optimizer.zero_grad()
                ims = ims.cuda()
                labels = labels.type(torch.long).cuda()
                preds = self.model(ims)
                loss = self.criterion(preds, labels)
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
