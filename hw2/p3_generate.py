import os
import argparse
import glob
# from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from model import DANN
from PIL import Image
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
import random


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class DANN(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(DANN, self).__init__()
        
        self.feature_extractor_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True)
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, n_class),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)

        return feature


    def forward(self, x, alpha=1.0):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output

    
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    same_seeds(1126)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = DANN().cuda()
    # print(model)

    state = torch.load(config.ckp_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.save_path
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_name,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn).convert('RGB')
                # print(data)
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.cuda()
                output, _ = model(data, 1)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw2_data/digits/mnistm/test')
    parser.add_argument('--target', type=str, default='mnistm')
    parser.add_argument('--save_path', type=str, default='ckpt/test/pred.csv')

    parser.add_argument('--ckp_path', default='ckpt/test/14000-dann_mu2.pth', type=str, help='Checkpoint path.')

    model_path = {
        "usps":"./p3_models/14000-dann_mu2.pth",
        "mnistm":"./p3_models/14000-dann_sm2.pth",
        "svhn": "./p3_models/14000-dann_us2.pth",
    }
    config = parser.parse_args()
    config.ckp_path = model_path[config.target]
    # parameter = {
    #     "img_dir": '/content/hw2_data/digits/svhn/test',
    #     "save_path": '/content/ckpt/test',
    #     "ckp_path": "/content/hw2_2_models/14000-dann_us2.pth",
    # }
    # config = Namespace(**parameter)
    print(config)
    main(config)