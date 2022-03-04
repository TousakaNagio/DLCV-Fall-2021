import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch
import random
from argparse import Namespace
import argparse



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

same_seeds(1126) #best1126

opt = Namespace(
    latent_dim = 100,
    n_classes = 10, 
    img_size = 32, 
    channels = 3, 
)


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def sample_image(n_row, generator):
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    transform = transforms.Resize(28)
    gen_imgs = transform(gen_imgs)
    return gen_imgs, labels   

def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    # print(args.model_path)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.model_path))

    imgs, labels = [], []
    for i in range(10):
        img, label = sample_image(10, generator)
        imgs.append(img)
        labels.append(label)

    imgs = torch.cat(imgs, dim = 0)
    labels = torch.cat(labels, dim = 0)
    imgs = imgs.to('cpu')
    labels = labels.to('cpu').numpy()
    index = [1] * 10
    
    for idx, img in enumerate(imgs):
        cur = labels[idx]
        save_image(img, os.path.join(args.save_dir, f'{cur}_{index[cur]:03d}.png'), normalize = True)
        index[cur] += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default = "./p2_generate", type = str)
    parser.add_argument("--model_path", type = str)
    args = parser.parse_args()
    main(args)

#collaborator thanks to b08902134 曾楊哲
