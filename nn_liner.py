import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=Linear(in_features=196608,out_features=10)


    def forward(self,input):
        output=self.linear1(input)
        return output

model=Model()
#将196608变成10
for data in dataloader:
    imgs,targets=data
    #数据传入模型前需要格式化，具体是将矩阵变换成一条直线
    #方法一
    #input=torch.reshape(imgs,(1,1,1,-1))
    #方法二
    input=torch.flatten(imgs)
    output=model(input)
    print(output)