import torch
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d,Flatten
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

image_path="imgs/bird.png"
image=Image.open(image_path)
image=image.convert('RGB')
tranform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                         torchvision.transforms.ToTensor()])
image=tranform(image)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            Flatten(),
            Linear(1024,64),
            Linear(64,10),
        )

    def forward(self,x):
        x=self.model1(x)
        return x

#model = Model()#.to(device)
#model=torch.load("model_50.pth")
model=torch.load("../model/model_vgg16_20.pth",map_location=torch.device("cpu"))
#model=torch.load("../model/model_50.pth",map_location=torch.device("cpu"))
image=image#.to(device)
image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1).item())