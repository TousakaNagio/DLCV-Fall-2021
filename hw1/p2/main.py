import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace

from utils import same_seeds
from dataset import Dataset
from Train import Train



def main(config):

    same_seeds(1126)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_set = Dataset(root=config.trainset_dir, transform=transform, randomflip=True)
    valid_set = Dataset(root=config.valset_dir, transform=transform)

    trainset_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=1)
    validset_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=1)
    
    solver = Train(trainset_loader, validset_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # # Training configuration.
    # parser.add_argument('--trainset_dir', type=str, default='../p2_data/train')
    # parser.add_argument('--valset_dir', type=str, default='../p2_data/validation')
    # parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--epoch', type=int, default=100, help='number of total epoch')
    # parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this iteration')
    # parser.add_argument('--name', default='', type=str, help='Name for saving model')
    # parser.add_argument('--model_type', default='FCN32s', choices=['FCN32s', 'FCN16s', 'FCN8s'], type=str, help='Model type')
    # parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    # parser.add_argument('--log_interval', type=int, default=50)
    # parser.add_argument('--save_interval', type=int, default=200)
    
    parameter = {
        'trainset_dir' : '/content/p2_data/train',
        'valset_dir' : '/content/p2_data/validation',
        'batch_size' : 4,
        'lr' : 1e-4,
        'epoch' : 100,
        'resume_iter' : 0,
        'name' : 'test',
        'model_type' : 'DeepLabv3_ResNet50',
        'ckp_dir' : 'ckpt/',
        'log_interval' : 50,
        'save_interval' : 200,
    }
    parameter = Namespace(**parameter)
    # config = parser.parse_args()
    # print(config)
    main(parameter)