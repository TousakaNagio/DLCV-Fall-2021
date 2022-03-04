import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup

from dataset import IMAGE
from model import Net
"requirement !pip install transformers==4.5.0"

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization = False)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train_save(model, epoch, save_interval, log_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    max_steps = len(train_loader) / batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, 15, max_steps)
    
    iteration = 0
    for ep in range(epoch):
        if ep%5 == 0:
          scheduler.step()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Lr: {}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('/content/checkpoint/model%i.pth' % iteration, model, optimizer)
            iteration += 1
        validate(model)
    save_checkpoint('/content/checkpoint/model%i.pth' % iteration, model, optimizer)

def validate(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


same_seeds(0)


train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(p=0.3),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

valid_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

train_set = IMAGE(root = '/content/p1_data/train_50',transform = train_tfm)
valid_set = IMAGE(root = '/content/p1_data/val_50',transform = valid_tfm)

# a,b = valid_set[68]
# print(b)

# print(len(train_set))
# print(len(valid_set))

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# a,b = valid_set[0]
# print(a.shape)
# print(b)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Net().to(device)

print(model)

train_save(model, 50, 500)



