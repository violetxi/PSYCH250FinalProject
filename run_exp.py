import os
import pickle
import argparse
import numpy as np
import pandas as pd
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

LABEL_MAP = {0 : 'faces', 1 : 'bodies',
             2 : 'places', 3 : 'characters'}

def load_args():
    parser = argparse.ArgumentParser(description='Leison study on AlexNet')
    parser.add_argument('--ckpt_folder', type=str,
                        help='Where all the checkpoints are saved.')
    parser.add_argument('--result_folder', type=str,
                        help='Where to save trained models.')
    parser.add_argument('--meta_path', type=str,
                        help="Where processed PT files are.")
    parser.add_argument('--result_by_label', action='store_true',
                        help='Whether to store results by label')
    args = parser.parse_args()
    return args


class RunExp:
    def __init__(self, args, n_exps, lesion_percent=0.2, seed=0):
        np.random.seed(seed)    # to reproduce the randomly sampled results
        self.args = args
        self.ckpt_paths = self.get_all_checkpoint_paths()
        self.lesion_percent = lesion_percent
        self.n_exps = n_exps
        self.result_by_label = self.args.result_by_label
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
            dataset, batch_size=1, shuffle=False, num_workers=0)

    def eval_model(self, untrained=False):
        if untrained:
            self.model = Model(pretrained=False)
            self.model.cuda()
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
        print('Running lesion experiments')
        for ckpt_path in tqdm(self.ckpt_paths):            
            self.load_model(ckpt_path)
            epoch = os.path.basename(ckpt_path).split('.')[0]
            if self.result_by_label:
                result_path = os.path.join(
                    self.args.result_folder,
                    f'lesion_exp_category_epoch_{epoch}.pkl'
                )
            else:
                result_path = os.path.join(
                    self.args.result_folder,
                    f'lesion_exp_overall_epoch_{epoch}.csv'
                )
            results = {}
            index = []
            for layer in tqdm(self.target_layers):             
                index.append(layer)
                for n_exp in range(self.n_exps):
                    exp = f'exp_{n_exp}'
                    self.model_state_dict = self.model.state_dict()    # keep tracking of modified state dict
                    self.lesion_one_layer(layer)
                    self.model.load_state_dict(self.model_state_dict)
                    if self.result_by_label:
                        accuracy = self.eval_model_by_label()
                    else:
                        accuracy = self.eval_model()
                    if exp not in results:
                        results[exp] = [accuracy]
                    else:
                        results[exp].append(accuracy)
                
            self.store_results(results, result_path, index)
                    
    def eval_ckpts(self):
        print('evaluating checkpoints')
        if self.result_by_label:
            accuracy = self.eval_model_by_label(untrained=True)
            result_path = f'{self.args.result_folder}/epoch_results_by_label.csv'
            results = {}
            results['untarined'] = accuracy
        else:
            accuracy = self.eval_model(untrained=True)
            result_path = f'{self.args.result_folder}/epoch_results_overall.csv'
            results = {'epoch': [], 'accuracy': []}        
            results['epoch'].append('untrained')
            results['accuracy'].append(accuracy)
        
        for ckpt_path in tqdm(self.ckpt_paths):
            self.load_model(ckpt_path)
            epoch = int(os.path.basename(ckpt_path).split('.')[0])
            if self.result_by_label:
                accuracy = self.eval_model_by_label()
                results[epoch] = accuracy
            else:
                accuracy = self.eval_model()            
                results['epoch'].append(epoch)
                results['accuracy'].append(accuracy)

        self.store_results(results, result_path)

    def eval_model_by_label(self, untrained=False):
        if untrained:
            self.model = Model(pretrained=False)
            self.model.cuda()
        label_result = {
            label : {'correct': 0, 'total': 0} for label in LABEL_MAP
        }
        correct = 0
        total = 0
        for i, (ims, labels) in enumerate(self.dataloader):
            ims = ims.cuda()            
            labels = labels.cuda()
            outputs = self.model(ims)
            preds = torch.argmax(outputs, 1)
            label_result[labels.item()]['total'] += preds.size()[0]
            correct = (preds == labels).sum().item()
            label_result[labels.item()]['correct'] += correct

        final_label_result = {LABEL_MAP[label] : label_result[label]['correct'] /
                              label_result[label]['total'] for label in label_result}
        return final_label_result
        
    def store_results(self, results, path, index=None):
        print(path)
        if self.result_by_label and path.endswith('pkl'):
            pickle.dump(results, open(path, 'wb'))
        else:
            if index:
                df = pd.DataFrame(results, index=index)
            else:
                df = pd.DataFrame(results)
            df.to_csv(path)

    def do_exps(self):
        self.eval_ckpts()
        self.run_lesion_exps()
    
    
if __name__ == '__main__':
    args = load_args()
    n_random_lesions = 10
    run_exp = RunExp(args, n_random_lesions)
    run_exp.do_exps()
