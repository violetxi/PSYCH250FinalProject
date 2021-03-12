import os
import argparse
import numpy as np
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
    parser.add_argument('--meta_path', type=str,
                        help="Where processed PT files are.")
    args = parser.parse_args()
    return args


class RunExp:
    def __init__(self, args, n_random_lesions, lesion_percent=0.2, seed=0):
        np.random.seed(seed)    # to reproduce the randomly sampled results
        self.args = args
        self.ckpt_paths = self.get_all_checkpoint_paths()
        self.lesion_percent = lesion_percent
        self.n_random_lesions = n_random_lesions
        # 5 conv layers to be lesioned
        self.target_layers = [
            'model.features.0', 'model.features.3', 'model.features.6',
            'model.features.8', 'model.features.10']
        self.load_data()

    def get_all_checkpoint_paths(self):
        ckpt_folder = self.args.ckpt_folder
        ckpt_files = sorted(
            os.listdir(ckpt_folder),
            key=lambda x: int(os.path.splitext(x)[0]))
        return [os.path.join(
            ckpt_folder, ckpt_file) for ckpt_file in ckpt_files]        
        
    def load_model(self, ckpt_path):
        self.model = Model(pretrained=False)    # not loading pre-trained Alexnet because loading our own
        self.model.cuda()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()        
        
    def load_data(self):
        dataset = StimuliDataset(self.args.meta_path, train=False)
        self.dataloader = DataLoader(
            dataset, batch_size=200, shuffle=False, num_workers=0)

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

    def get_lesion_index(self, num_channels):        
        num_lesions = int(num_channels * self.lesion_percent)
        all_idxs = np.arange(num_channels)
        return np.random.choice(all_idxs, num_lesions)
        
    def lesion_one_layer(self, layer):
        weight_key = layer + '.weight'
        bias_key = layer + '.bias'
        weights = self.model_state_dict[weight_key]
        bias = self.model_state_dict[bias_key]
        lesion_locs = self.get_lesion_index(bias.shape[0])
        weights[lesion_locs, :, :, :] = 0.0
        bias[lesion_locs] = 0.0        

    # lesion experiments are only done on the imgnt-pretrained or trained net
    def run_lesion_exps(self):
        for ckpt_path in self.ckpt_paths:
            self.load_model(ckpt_path)            
            for layer in self.target_layers[:2]:
                self.model_state_dict = self.model.state_dict()    # keep tracking of modified state dict
                self.lesion_one_layer(layer)
                self.model.load_state_dict(self.model_state_dict)
                accuracy = self.eval_model()
                print(ckpt_path, layer, accuracy)
    
    def eval_ckpts(self):        
        for ckpt_path in self.ckpt_paths:
            self.load_model(ckpt_path)
            accuracy = self.eval_model()
            print(f'{ckpt_path} validataion accuracy: {accuracy}')
    

if __name__ == '__main__':
    args = load_args()
    n_random_lesions = 10
    run_exp = RunExp(args, n_random_lesions)
    #run_exp.eval_ckpts()
    run_exp.run_lesion_exps()
