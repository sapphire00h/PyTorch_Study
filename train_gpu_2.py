import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d,Flatten
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

#定义训练设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#准备数据集
train_data=torchvision.datasets.CIFAR10("../dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                     download=True)
test_data=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
train_data_size=len(test_data)
test_data_size=len(test_data)
print("训练数据集长度：{}\n测试数据集长度：{}".format(train_data,test_data))

#加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#搭建神经网络

#创建网络模型
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


model=Model()
model.to(device)
#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn.to(device)
#定义优化器
learning_rate=0.001
#1e-2=1*10^(-2)=1/100=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#设置训练网络参数

#记录训练次数
total_train_step=0
#记录测试次数
total_test_step=0
#训练轮数
epoch=10

#添加Tensorboard
#writer=SummaryWriter("../logs_train")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    model.train()
    #训练步骤开始
    for data in train_dataloader:
        imgs,targets=data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=model(imgs)
        #计算loss
        loss=loss_fn(outputs,targets)
        #清零梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #进行优化
        optimizer.step()
        if total_train_step%100==0:
            print("训练次数{},loss:{}".format(total_train_step,loss.item()))
            #writer.add_scalar("train_loss",loss.item(),total_train_step)
        total_train_step = total_train_step + 1

    #测试步骤开始，取消梯度计算
    model.eval()
    total_test_loss=0
    total_accuracy=0
    #对现有模型进行测试
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            #loss是tensor数据类型，加上item()
            total_test_loss=total_test_loss+loss.item()
            #计算正确率
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy

        print("整体测试集上的loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
        #writer.add_scalar("test_loss",total_test_loss,total_test_step)
        #writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
        total_test_step=total_test_step+1

        #保存训练模型
        if total_train_step%5==0:
            torch.save(model,"model_{}.pth".format(total_test_step))
            #torch.save(model.state_dict(), "model_{}.pth".format(total_test_step))
            print("第{}轮训练的型已保存".format(total_test_step))

#writer.close()