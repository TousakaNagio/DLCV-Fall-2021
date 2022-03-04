import os
import argparse
import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader
from argparse import Namespace
import random
import numpy as np

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

def main(args):
    same_seeds(7414)

    loader = ModelLoader(
        base_dir = '.',   
        name = 'default',
        load_from = args.load_from
    )

    os.makedirs(args.output, exist_ok = True)
    
    for i in range(1000):
        noise   = torch.randn(1, 512).cuda() 
        styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  
        images  = loader.styles_to_images(styles) 

        save_image(images, os.path.join(args.output, f'{i}.jpg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default = "./output_image", type = str)
    parser.add_argument("--load_from", default = -1, type = int)
    args = parser.parse_args()
    # parameter = {
    #     "output": '/content/generate1/',
    #     "model": 68,
    # }
    # args = Namespace(**parameter)
    main(args)

#collaborator thanks to b08902134 曾楊哲
