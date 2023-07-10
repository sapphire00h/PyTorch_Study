import  torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试数据集
test_data=torchvision.datasets.CIFAR10(root="../dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

'''
batch_size=4,每次取出4个数据集进行打包
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

drop_last=True
图片数量不足时候会自动舍去
'''
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

#测试数据集中第一张图片及target
img,target=test_data[0]
print(img.shape)
print(target)
"""
数据是三通道图片
torch.Size([3, 32, 32])
3
"""



"""
#查看test_loader中每一个打包好的batch
for data in test_loader:
    imgs,targets=data
    print(imgs.shape)
    print(targets)
    
torch.Size([4, 3, 32, 32])
torch.Size([4张图片（每个batch中的图片数量）, 3通道, 32, 32（大小是32*32）])
"""
writer=SummaryWriter("dataloader")
#设置初始步长
step=0
for data in test_loader:
    imgs, targets = data
    #图片很多，所以使用add_images方法
    writer.add_images("test_data__drop_last=True",imgs,step)
    step=step+1

writer.close()