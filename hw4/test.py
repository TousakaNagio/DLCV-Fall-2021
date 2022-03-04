import torch

device = torch.device('cuda' if self.use_cuda else 'cpu')
print(device)