import os
import pdb
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class StimuliDataset(Dataset):
    def __init__(self, root_dir, train):
        self.train = train
        self.root_dir = root_dir
        self.__load_data()
        self.__build_transform()

    def __build_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                                         
            std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),         
            transforms.ToTensor(),            
            normalize,
        ])
        
    def __load_data(self):
        if self.train:
            path = os.path.join(self.root_dir, 'train.pt')
        else:
            path = os.path.join(self.root_dir, 'val.pt')            
        self.data, self.label = torch.load(path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        label = self.label[idx]
        return data, label
            
if __name__ == '__main__':
    root_dir = 'data/processed/'
    train_set = StimuliDataset(root_dir, True)
    val_set = StimuliDataset(root_dir, False)
    
