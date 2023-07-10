import torch
import torchvision
from tensorboardX import SummaryWriter
from theano.tensor import Flatten
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

# dataset=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
#                                      download=True)
# dataloader=DataLoader(dataset,batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        '''
        self.conv1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        self.maxpool1=MaxPool2d(kernel_size=2)
        self.conv2=Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3=Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.maxpool3=MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.linear1=Linear(in_features=1024,out_features=64)
        self.linear2=Linear(in_features=64,out_features=10)
        '''
        self.model1=nn.Sequential(
            Conv2d(3,32,5,padding=2),
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

        '''
        x=self.conv1(x)
        x=self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        '''



model=Model()
print(model)
#测试网络
input=torch.ones((64,3,32,32))
output=model(input)
print(output)
print(output.shape)

writer=SummaryWriter("logs_seq")
writer.add_graph(model,input)
writer.close()