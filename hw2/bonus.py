import os
import argparse
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.autograd import Variable, Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class myDataset(Dataset):
    def __init__(self, root, label_data=None, transform=None):
        self.transform = transform
        self.label_data = label_data

        self.img_paths = []
        self.labels = []

        if label_data is not None:
            for d in self.label_data:
                img_path, label = d.split(',')
                self.img_paths.append(os.path.join(root, img_path))
                self.labels.append(int(label))
        else:
            for fn in glob.glob(os.path.join(root, '*.png')):
                self.img_paths.append(fn)
                self.labels.append(0)

        self.len = len(self.img_paths)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.img_paths[index], self.labels[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class DSN(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential(
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

        self.source_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True),
        )

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential(
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

        self.target_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True),
        )

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential(
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

        self.shared_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True)
        )

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, n_class)
        )

        # classify two domain
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2)
        )

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(code_size, 50 * 4 * 4),
            nn.ReLU(True)
        )

        self.shared_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(50, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, bias=False),
            nn.Tanh()
        )
        

    def encode(self, input_data):
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 50 * 4 * 4)
        shared_code = self.shared_encoder_fc(shared_feat)
        return shared_code
    

    def forward(self, input_data, mode, rec_scheme='all', p=0.0):
        if mode == 'source':
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 50 * 4 * 4)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 50 * 4 * 4)
            private_code = self.target_encoder_fc(private_feat)

        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 50 * 4 * 4)
        shared_code = self.shared_encoder_fc(shared_feat)

        class_label = self.shared_encoder_pred_class(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        recon = self.shared_decoder_fc(union_code)
        recon = recon.view(-1, 50, 4, 4)
        recon = self.shared_decoder_conv(recon)
        return class_label, domain_label, private_code, shared_code, recon


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    model = DSN().cuda()

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
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.cuda()
                output, _, _, _, _ = model(data, mode=config.mode)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw2_data/digits/svhn/test')
    parser.add_argument('--target', type=str, default='mnistm')
    parser.add_argument('--save_path', type=str, default='ckpt/test')
    parser.add_argument('--mode', default='target', type=str, help='mode for model')
    parser.add_argument('--ckp_path', default='ckpt/test/14000-dann_mu2.pth', type=str, help='Checkpoint path.')

    model_path = {
        "usps":"./bonus_models/29000-dsn_mu.pth",
        "mnistm":"./bonus_models/29000-dsn_sm.pth",
        "svhn": "./bonus_models/29000-dsn_us.pth",
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