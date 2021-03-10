import os
import pdb
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class StimuliDataset(Dataset):
    def __init__(self, meta_path, train):
        self.train = train        
        metas = pickle.load(open(meta_path, 'rb'))
        self.image_metas = metas['train'] if self.train else metas['test']
        self.__build_transform()

    def __build_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                                         
            std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),         
            transforms.ToTensor(),            
            normalize,
        ])        
        
    def __len__(self):
        return len(self.image_metas)

    def __getitem__(self, idx):
        image_path, label = self.image_metas[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(label)
        return image, label
            
if __name__ == '__main__':
    root_dir = 'data/processed/'
    train_set = StimuliDataset(root_dir, True)
    val_set = StimuliDataset(root_dir, False)
    
