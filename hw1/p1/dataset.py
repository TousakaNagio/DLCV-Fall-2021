import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class IMAGE(Dataset):
  def __init__(self,root,transform=None):
    self.transform = transform
    self.filenames = []
    files = sorted(glob.glob(os.path.join(root, '*.png')), key = lambda x:(int(x.split('/')[-1].split('_')[0]), int(x.split('_')[-1].split('.')[0])))
    # files = glob.glob(os.path.join(root, '*.png'))
    # files = listdir(root)
    for fn in files:
      head = fn.split('/')[4]
      head = head.split('_')[0]
      head = int(head)
      self.filenames.append((fn, head))
    self.len = len(self.filenames)

  def __getitem__(self,index):
    image_fn,label = self.filenames[index]
    image = Image.open(image_fn)
    if self.transform is not None:
      image = self.transform(image)
    return image,label

  def __len__(self):
    return self.len