import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)    # 4 classes in fLoc dataset

    def forward(self, x):
        x = self.model(x)
        return x
