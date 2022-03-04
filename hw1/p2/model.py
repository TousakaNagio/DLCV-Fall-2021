import torch.nn as nn
import torchvision.models as models


class FCN32s(nn.Module):
    def __init__(self, num_classes = 7):
        super(FCN32s, self).__init__()
        self.name = "vgg16_FCN32s"
        self.num_classes = num_classes
        self.vgg = models.vgg16(pretrained=True)
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.score = nn.Conv2d(4096, self.num_classes, 1),
        self.upscore = nn.ConvTranspose2d(self.num_classes, self.num_classes, 64, stride = 32, bias = False)

    def forward(self, x):
        x_shape = x.shape
        x = self.vgg.features(x)
        x = self.fcn(x)
        x = self.score(x)
        x = self.upscore(x)
        x = x[:, :, 16: 16 + x_shape[2], 16: 16 + x_shape[3]]
        return x

# Reference: https://github.com/kai860115/DLCV2020-FALL/blob/main/hw2/semantic_segmentation/model.py

class DeepLabv3_ResNet50(nn.Module):
    def __init__(self, num_classes = 7):
        super(DeepLabv3_ResNet50, self).__init__()
        self.name = "DeepLabv3_ResNet50"
        self.model = models.segmentation.deeplabv3_resnet50(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)['out']
        return x