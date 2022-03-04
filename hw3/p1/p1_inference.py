import os
import argparse
import glob
import torch
import torchvision.models as models
from torch import nn
import torchvision.transforms as transforms
# from model import Net
from PIL import Image
from pytorch_pretrained_vit import ViT


class Net(nn.Module):
    def __init__(self):
      # TODO
      super(Net, self).__init__()

      # self.load_model = models.wide_resnet50_2(pretrained = True, progress=True)
      self.load_model = ViT('B_16_imagenet1k', pretrained=True)
      # print(self.load_model.image_size)
      self.Layer = nn.Linear(1000, 37)
      # self.resnet18.fc.out_features = 50

    def forward(self, x):
      # TODO 
      x = self.load_model(x)
      x = self.Layer(x)
      return x


def main(config):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = Net().to(device)

    state = torch.load(config.model_path)
    model.load_state_dict(state['state_dict'])

    # os.makedirs(config.save_dir, exist_ok = True)
    os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)
    out_filename = config.save_dir
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = torch.unsqueeze(transform(Image.open(fn).convert('RGB')), 0)
                output = model(data.to(device))
                pred = output.max(1, keepdim=True)[1]
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='./hw3_data/p1_data/val_50')
    parser.add_argument('--save_dir', type=str, default='../')
    parser.add_argument('--model_path', default='./model2500.pth', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    print(config)
    main(config)

# reference: https://github.com/kai860115/DLCV2020-FALL/blob/main/hw2/image_classification/test.py
