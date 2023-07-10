import torch
from torch import nn

#简单神经网络创建
class Model(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,input):
        output=input+1
        return  output

Mod=Model()
x=torch.tensor(1.0)
output=Mod(x)
print(output)