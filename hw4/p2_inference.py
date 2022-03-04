import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim

import csv
import random
import numpy as np
import pandas as pd
from argparse import Namespace

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

import torch
# from byol_pytorch import BYOL
from torchvision import models

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

# mini-Imagenet dataset
class OfficeDataset(Dataset):
    def __init__(self, csv_path, data_dir, transforms):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.labels = list(sorted(set(pd.read_csv(csv_path).iloc[:,2])))
        self.transform = transforms
        
        self.label_dict = {}
        # for id, lab in enumerate(self.labels):
        #   self.label_dict[lab] = id
        
    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label_str = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        label = self.label_dict[label_str]
        return image, label

    def __len__(self):
        return len(self.data_df)

class Net(nn.Module):
    def __init__(self):
      # TODO
      super(Net, self).__init__()
      self.resnet = models.resnet50(pretrained=False).to('cuda')
      self.classifier = nn.Linear(1000, 65)

    def forward(self, x):
      # TODO
      x = self.resnet(x)
      x = self.classifier(x)
      return x

def get_key(my_dict, val):
    for key, value in my_dict.items():
          if val == value:
              return key

    return "key doesn't exist"

def main(args):
    val_transform = transforms.Compose([
        filenameToPILImage,
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    test_set = OfficeDataset(args.val_csv, args.val_path, transforms = val_transform)
    labels_dict = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
    test_set.label_dict = labels_dict
    batch_size = 128
    valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    state = torch.load(args.model_path)
    model.load_state_dict(state['state_dict'])


    output_csv = args.out_path
    val_csv = args.val_csv
    names = pd.read_csv(val_csv).iloc[:,1]

    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(pred)):
              preds.append(int(pred[i]))
            correct += pred.eq(target.view_as(pred)).sum().item()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    with open(output_csv, 'w') as out_file:
        out_file.write('id,filename,label\n')
        for i in range(len(preds)):
          out_file.write('{},{},{}\n'.format(i, names[i], get_key(test_set.label_dict, preds[i])))

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))



def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    # parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    # parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    # parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    # parser.add_argument('--M_aug', default=10, type=int, help='M_augmentation (default: 10)')
    # parser.add_argument('--matching_fn', default='l2', type=str, help='distance matching function')
    # parser.add_argument('--load', type=str, help="Model checkpoint path")
    # parser.add_argument('--type', type=str, choices=['proto', 'dhm', 'improved'], help="Model type")
    # parser.add_argument('--test_csv', default='./hw4_data/val.csv', type=str, help="Testing images csv file")
    # parser.add_argument('--test_data_dir', default='./hw4_data/val', type=str, help="Testing images directory")
    # parser.add_argument('--testcase_csv', default='./hw4_data/val_testcase.csv', type=str, help="Test case csv")
    # parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--val_csv', default='/content/hw4_data/office/val.csv', type=str)
    parser.add_argument('--val_path', default='/content/hw4_data/office/val', type=str)
    parser.add_argument('--out_path', default='/content/output.csv', type=str)
    parser.add_argument('--model_path', default='/content/model2790.pth', type=str)

    return parser.parse_args()
    # parameter = {
    #     'val_csv': '/content/hw4_data/office/val.csv',
    #     'val_path': '/content/hw4_data/office/val',
    #     'out_path': '/content/output.csv',
    #     'model_path': '/content/model2046.pth',
    # }
    # config = Namespace(**parameter)
    # return config

if __name__=='__main__':
    args = parse_args()
    main(args)
