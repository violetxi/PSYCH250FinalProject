import torch

checkpoint = torch.load('checkpoints/90.pth')
print(checkpoint['losses'])
