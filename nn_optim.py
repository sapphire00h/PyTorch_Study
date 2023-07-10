import torch
import torchvision
from tensorboardX import SummaryWriter
from theano.tensor import Flatten
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

#加载数据集，转为tensor数据类型
dataset=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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

loss=nn.CrossEntropyLoss()
model=Model()
optim=torch.optim.SGD(model.parameters(),lr=0.001)

for epoch in range(10):
    running_loss=0.0
    # 将所有数据进行一次循环计算梯度并更新参数
    for data in dataloader:
        imgs, targets = data
        outputs = model(imgs)
        # 计算loss
        result_loss = loss(outputs, targets)
        # 先将优化器的参数归零
        optim.zero_grad()
        # 使用result_loss进行反向传播
        result_loss.backward()
        optim.step()
        running_loss=running_loss+result_loss

    print(running_loss)







#测试网络
'''
print(model)
input=torch.ones((64,3,32,32))
output=model(input)
print(output)
print(output.shape)
'''

