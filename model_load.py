import torch
import torchvision

#加载方式一
model=torch.load("vgg16_method1.pth")
#print(model)
#加载方式二
model2=torchvision.models.vgg16(pretrained=False)
model2.load_state_dict(torch.load("vgg16_method2.pth"))
print(model2)
