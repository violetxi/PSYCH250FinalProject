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
    parser.add_argument('--ckpt_folder', type=str,
                        help='Where all the checkpoints are saved.')
    parser.add_argument('--result_folder', type=str,
                        help='Where to save trained models.')
    parser.add_argument('--data_root', type=str,
                        help="Where processed PT files are.")
    args = parser.parse_args()
    return args


class RunExp:
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.load_data()

    def load_model(self):
        self.model = Model()
        self.model.cuda()
        
    def load_data(self):
        dataset = StimuliDataset(self.args.data_root, train=False)
        self.dataloader = DataLoader(
            dataset, batch_size=200,
            shuffle=False, num_workers=4)

    def eval_model(self):
        correct = 0
        total = 0
        for i, (ims, labels) in enumerate(self.dataloader):
            ims = ims.cuda()
            save_image(ims, f'val_{i}.png')
            labels = labels.cuda()
            outputs = self.model(ims)
            preds = torch.argmax(outputs, 1)
            total += preds.size()[0]
            correct += (preds == labels).sum().item()
        return correct / total
        
    def eval_ckpts(self):
        ckpt_folder = self.args.ckpt_folder
        ckpt_files = os.listdir(ckpt_folder)
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(ckpt_folder, ckpt_file)
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()            
            accuracy = self.eval_model()
            print(f'{ckpt_file} validataion accuracy: {accuracy}')
        

if __name__ == '__main__':
    args = load_args()
    run_exp = RunExp(args)
    run_exp.eval_ckpts()
