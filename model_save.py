import torch
import torchvision
'''
当pretrained参数设置为True时，torchvision.models.vgg16(pretrained=True)会自动下载并加载预训练的权重。这意味着你可以立即使用已在大型图像数据集上进行训练的VGG16模型，而不需要自己从头开始进行训练。这对于许多计算机视觉任务来说是非常有用的，因为预训练的模型通常具有较好的特征提取能力。
'''
vgg16=torchvision.models.vgg16(pretrained=False)

#方法一
torch.save(vgg16,"vgg16_method1.pth")

#方式二,将vgg16中的参数保存成字典
#torch.save(vgg16.state_dict(),"vgg16_method2.pth")