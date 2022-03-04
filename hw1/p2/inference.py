import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from model import DeepLabv3_ResNet50, FCN32s
from PIL import Image


mask = {
    0: (0, 1, 1),
    1: (1, 1, 0),
    2: (1, 0, 1),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (1, 1, 1),
    6: (0, 0, 0),
}


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DeepLabv3_ResNet50().to(device)

    state = torch.load(config.model_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)

    os.makedirs(config.save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for fn in filenames:
            ImageID = fn.split('/')[-1].split('_')[0]
            output_filename = os.path.join(config.save_dir, '{}_mask.png'.format(ImageID))  
            data = transform(Image.open(fn))
            data_shape = data.shape
            data = torch.unsqueeze(data, 0)
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1].reshape((-1, data_shape[1], data_shape[2])) # get the index of the max log-probability
            y = torch.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]))
            for k, v in mask.items():
                y[:,0,:,:][pred == k] = v[0]
                y[:,1,:,:][pred == k] = v[1]
                y[:,2,:,:][pred == k] = v[2]

            y = transforms.ToPILImage()(y.squeeze())
            y.save(output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../hw1_data/p2_data/validation')
    parser.add_argument('--save_dir', type=str, default='../output')
    parser.add_argument('--model_path', default='../ckpt/p2_model_resnet50.ckpt', type=str, help='Checkpoint path.')
    

    config = parser.parse_args()
    print(config)
    main(config)

# Reference: thanks to https://github.com/kai860115/DLCV2020-FALL/blob/main/hw2/semantic_segmentation/test.py
