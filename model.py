import torch
import torchvision
from tensorboardX import SummaryWriter
from theano.tensor import Flatten
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

#搭建神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1=nn.Sequential(
            Conv2d(3,32,5,stride=1,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            nn.Flatten(),
            Linear(1024,64),
            Linear(64,10),
        )

    def forward(self,x):
        x=self.model1(x)
        return x


if __name__ =='__main__':
    #验证网络正确性
    model=Model()
    input=torch.ones((64,3,32,32))
    output=model(input)
    print(output.shape)