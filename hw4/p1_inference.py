import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from argparse import Namespace

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Protonet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.conv = Convnet(in_channels, hid_channels, out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 400)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def pairwise_distances(x, y, matching_fn='parametric', parametric=None):
    n_x = x.shape[0]
    n_y = y.shape[0]
    x_exp = x.unsqueeze(1).expand(n_x, n_y, -1).reshape(n_x*n_y, -1)
    y_exp = y.unsqueeze(0).expand(n_x, n_y, -1).reshape(n_x*n_y, -1)
    
    distances = parametric(torch.cat([x_exp, y_exp], dim=-1))
    
    return distances.reshape(n_x, n_y)

def predict(args, model, data_loader):
    for _, m in model.items():
        m.eval()

    prediction_results = []
    episodic_acc = []


    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            data = data.cuda()
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            # label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            #  extract the feature of support and query data
            support = model['proto'](support_input)
            queries = model['proto'](query_input)

            #  calculate the prototype for each class according to its support data
            prototypes = support.reshape(args.N_way, args.N_shot, -1).mean(dim=1)
            distances = pairwise_distances(queries, prototypes, args.matching_fn, model['parametric'])

            #  classify the query data depending on the its distense with each prototype
            y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
            prediction_results.append(y_pred.reshape(-1))

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--M_aug', default=10, type=int, help='M_augmentation (default: 10)')
    parser.add_argument('--matching_fn', default='parametric', type=str, help='distance matching function')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', default='/content/hw4_data/mini/val.csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', default='/content/hw4_data/mini/val', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', default='/content/hw4_data/mini/val_testcase.csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()
    # parameter = {
    #     'N_way': 5,
    #     'N_shot': 1,
    #     'N_query': 15,
    #     'M_aug': 10,
    #     'matching_fn': 'parametric',  #cosine parametric l2
    #     'load': '/content/15600-protonet.pth',
    #     'test_csv': '/content/hw4_data/mini/val.csv',
    #     'test_data_dir': '/content/hw4_data/mini/val',
    #     'testcase_csv': '/content/hw4_data/mini/val_testcase.csv',
    #     'output_csv': './output2.csv',
    # }
    # config = Namespace(**parameter)
    # return config

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    state = torch.load(args.load)
    model = {}
    model['proto'] = Protonet().cuda()
    model['proto'].load_state_dict(state['state_dict'])
    model['parametric'] = nn.Sequential(
        nn.Linear(800, 400),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(400, 1)
    ).cuda()
    model['parametric'].load_state_dict(state['parametric'])

    prediction_results = predict(args, model, test_loader)

    if os.path.dirname(args.output_csv):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    # TODO: output your prediction to csv
    with open(args.output_csv, 'w') as out_file:
        line = 'episode_id'
        for i in range(args.N_way*args.N_query):
            line += ',query%d' % (i)
        line += '\n'
        out_file.write(line)

        for i, prediction in enumerate(prediction_results):
            line = '%d' % (i)
            for j in prediction:
                line += ',%d' % j
            line += '\n'
            out_file.write(line)

#Reference: https://github.com/kai860115/DLCV2020-FALL/tree/main/hw4