import torchvision
from torch.utils.tensorboard import SummaryWriter

#Ctrl+P查看提示

#将数据集的每一张图片都转为Tensor数据类型，使用Compose方法
dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

print(test_set[0])
writer=SummaryWriter("CIFAR10")
for i in range(50):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()
'''
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

print(test_set[0])
print(test_set.classes)
#可以看到两个属性，第一个是图片属性，第二个是标签属性
img,target=test_set[0]
print(img)
print(target)
print("图片所属的类别："+test_set.classes[target])
img.show()
'''
