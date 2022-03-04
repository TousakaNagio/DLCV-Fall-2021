import torchvision.models as models
from torch import nn


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.load_model = models.wide_resnet50_2(pretrained = True)
      self.Layer = nn.Linear(1000, 50)
      # self.resnet18.fc.out_features = 50

    def forward(self, x):
      x = self.load_model(x)
      x = self.Layer(x)
      return x