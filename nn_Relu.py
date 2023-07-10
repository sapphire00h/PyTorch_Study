import torch
from torch import nn
from torch.nn import ReLU

input=torch.tensor([[1,-0.5],
                    [-1,3]])
input=torch.reshape(input,(-1,1,2,2))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #inplace=False在原来位置直接对结果进行替换，否则直接进行变换
        self.relu1=ReLU()

    def forward(self,input):
        output=self.relu1(input)
        return output

model=Model()
output=model(input)
print(output)
