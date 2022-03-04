import re, glob, os, random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

def mask_target(image):
    mask = transforms.ToTensor()(image)
    mask = 4 * mask[0] + 2 * mask[1] + 1 * mask[2]
    masks = torch.zeros(image.shape, dtype=torch.long)
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown
            
    return masks

class Dataset(Dataset):
    def __init__(self, path, transform=None, randomflip=False):
        self.transform = transform
        self.images, self.labels = None, None
        self.path = path
        self.randomflip = randomflip

        # read filenames
        sat_image = sorted(glob.glob(os.path.join(path, '*.jpg')))
        mask_image = sorted(glob.glob(os.path.join(path, '*.png')))

        for sat, mask in zip(sat_image, mask_image):
            self.filenames.append((sat, mask))
    
    def __len__(self):
        return len(self.filenames)
                
    def __getitem__(self, index):
        sat, mask = self.filenames[index]
        sat = Image.open(sat)
        mask = Image.open(mask)

        if self.randomflip:
            if random.random() > 0.5:
                sat = TF.hflip(sat)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                sat = TF.vflip(sat)
                mask = TF.vflip(mask)
            
        if self.transform is not None:
            sat = self.transform(sat)

        return sat, mask_target(mask)

# reference: https://github.com/kai860115/DLCV2020-FALL/blob/main/hw2/semantic_segmentation/dataset.py