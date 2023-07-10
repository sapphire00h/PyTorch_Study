import torchvision

# train_data=torchvision.datasets.ImageNet("../dataset",split="train",download=True,
#                                          transform=torchvision.transforms.ToTensor())
vgg16_false=torchvision.models.vgg16(pretrained=False)
#下载已经训练好的模型
#vgg16_true=torchvision.models.vgg16(pretrained=True)
#删除下载的预训练模型
'''
import torch
import os

# 获取缓存目录
cache_dir = torch.hub.get_dir()

# 删除VGG16预训练权重文件
weight_file_path = os.path.join(cache_dir, 'checkpoints', 'vgg16-397923af.pth')
print(weight_file_path)
os.remove(weight_file_path)
'''


