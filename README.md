# PyTorch
> Ctrl+F 搜索框
>
> Ctrl+P 自动提示需要什么函数

## 两大函数`dir()`



![image-20230607001610193](./PyTorch.assets/image-20230607001610193.png)

### `dir()`

```python
dir(torch.cuda.is_available())
Out[5]: 
['__abs__',
 '__add__',
......
 'real',
 'to_bytes']
```

### `help()`

```python
help(torch.cuda.is_available)
Help on function is_available in module torch.cuda:
is_available() -> bool
    Returns a bool indicating if CUDA is currently available.
```

## PyTorch加载数据

### Dataset

提供一种方式去获取数据及其label

- 如何获取每一个数据及其label
- 告诉我们数据的总量

### Dataloader

为网络提供不同的数据形式

## `TensorBoard`的使用

### `add_scalar`

```python
def add_scalar(
        self,
        tag,#图表的标题
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
):
        """Add scalar data to summary.

        Args:
            tag (str): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %

        """
```

![image-20230608173902367](./PyTorch.assets/image-20230608173902367.png)

### 打开`TensorBoard`
```shell
tensorboard --logdir=logs
```
显示完整的Step
```shell
tensorboard --logdir=dataloader --samples_per_plugin=images=10000
```

![image-20230608174410636](./PyTorch.assets/image-20230608174410636.png)

### `add_image`

```python
add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    Add image data to summary.
    
    Note that this requires the ``pillow`` package.
    
    Args:
        tag (str): Data identifier
        img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
        global_step (int): Global step value to record
        walltime (float): Optional override default walltime (time.time())
          seconds after epoch of event
        dataformats (str): Image data format specification of the form
          CHW, HWC, HW, WH, etc.
    Shape:
        img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
        convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
        Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
        corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
    
    Examples::
    
        from torch.utils.tensorboard import SummaryWriter
        import numpy as np
        img = np.zeros((3, 100, 100))
        img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    
        img_HWC = np.zeros((100, 100, 3))
        img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    
        writer = SummaryWriter()
        writer.add_image('my_image', img, 0)
    
        # If you have non-default dimension setting, set the dataformats argument.
        writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
        writer.close()
    
    Expected result:
    
    .. image:: _static/img/tensorboard/add_image.png
       :scale: 50 %

```

> 注意NumPy的图片格式
>
> ```python
> print(img_array.shape)
> (512, 768, 3)
> #与dataformats='HWC'对应
> writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
> ```

### `Normalize`

需要导入的包

```python
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
from PIL import Image
```

归一化公式

> *output[channel] = (input[channel] - mean[channel]) / std[channel]*
>
> 
>
> (input-0.5)/0.5=2*input-1
>
> input[0,1]
>
> result[-1,1]

```python
#归一化，图片是RGB三个信号通道，需要提供三个标准差
writer=SummaryWriter("logs")
print(img_tensor[0][0][0])

trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)

print(img_norm[0][0][0])
writer.add_image("mountain_norm",img_tensor)
writer.close()
```

## Useful-Transforms

### `Compose`使用

![image-20230613192418761](./PyTorch.assets/image-20230613192418761.png)

### `RandomCrop`随机裁剪

![image-20230613194327833](./PyTorch.assets/image-20230613194327833.png)

### Torchvision中的数据集使用

![image-20230706195302465](./PyTorch.assets/image-20230706195302465.png)

## 模型模板

```python
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
       
    def forward(self,input):
```

### 小实战

> 模型理解：将图片最后分为十个目标函数，因为这个数据集里面有十类物种的图片，就可以使用loss进行参数与目标标签的匹配

模型结构

![image-20230709134615153](./PyTorch.assets/image-20230709134615153.png)

### `3@32*32 -> 32@32*32`

![image-20230709134844293](./PyTorch.assets/image-20230709134844293.png)

![image-20230709135551232](./PyTorch.assets/image-20230709135551232.png)

```python
stride=1
padding=2
```

[随机梯度下降SGD]: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
[交叉熵损失]: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

### vgg16下载位置

![image-20230710130458162](./PyTorch.assets/image-20230710130458162.png)

## 模型训练思路

![image-20230710152926595](./PyTorch.assets/image-20230710152926595.png)

### 示例代码

```python
import torch
import torchvision
from tensorboardX import SummaryWriter
from theano.tensor import Flatten
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

#导入模型
from model import *

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
model=Model()
#损失函数
loss_fn=nn.CrossEntropyLoss()
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
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
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
        torch.save(model,"model_{}.pth".format(total_test_step))
        print("第{}轮训练的模型已保存".format(total_test_step))
        #torch.save(model.state_dict(), "model_{}.pth".format(total_test_step))

#writer.close()
```

### 具体步骤

#### 准备数据集

```python
#准备数据集
train_data=torchvision.datasets.CIFAR10("../dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                     download=True)
test_data=torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
train_data_size=len(test_data)
test_data_size=len(test_data)
print("训练数据集长度：{}\n测试数据集长度：{}".format(train_data,test_data))
```

#### 加载数据集

```python
#加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
```

#### 搭建神经网络

```python
#搭建神经网络
import torch
import torchvision
from tensorboardX import SummaryWriter
from theano.tensor import Flatten
from torch import nn
from torch.nn import Linear, Conv2d, MaxPool2d
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy

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
            nn.Flatten(),
            Linear(1024,64),
            Linear(64,10),
        )

    def forward(self,x):
        x=self.model1(x)
        return x

if __name__ =='__main__':
    #验证网络正确性
    model=Model()
    input=torch.ones((64,3,32,32))
    output=model(input)
    print(output.shape)
```

#### 创建网络模型、损失函数选择、优化器选择

```python
#创建网络模型
model=Model()
#损失函数
loss_fn=nn.CrossEntropyLoss()
#优化器
learning_rate=0.001
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
```

> #1e-2=1*10^(-2)=1/100=0.01

#### 设置训练网络参数、添加Tensorboard

```python
#设置训练网络参数

#记录训练次数
total_train_step=0
#记录测试次数
total_test_step=0
#训练轮数
epoch=10

#添加Tensorboard
#writer=SummaryWriter("../logs_train")
```

#### 开始训练与测试模型

```python
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    model.train()
    #训练步骤开始
    for data in train_dataloader:
        imgs,targets=data
        outputs=model(imgs)
        #计算loss
        loss=loss_fn(outputs,targets)
        #清零梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #进行优化
        optimizer.step()
        #可视化训练次数
        if total_train_step%100==0:
            print("训练次数{},loss:{}".format(total_train_step,loss.item()))
            #writer.add_scalar("train_loss",loss.item(),total_train_step)
        total_train_step = total_train_step + 1

    #在一轮训练后，测试步骤开始
    model.eval()
    total_test_loss=0
    total_accuracy=0
    #取消梯度计算
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            #loss是tensor数据类型，加上item()
            total_test_loss=total_test_loss+loss.item()
            #计算正确率
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
		
        #可视化此轮训练的整体loss和准确率
        print("整体测试集上的loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
        #将loss和准确率分别写入tensorboard
        #writer.add_scalar("test_loss",total_test_loss,total_test_step)
        #writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
        total_test_step=total_test_step+1

        #保存训练模型
        torch.save(model,"model_{}.pth".format(total_test_step))
        print("第{}轮训练的模型已保存".format(total_test_step))
        #torch.save(model.state_dict(), "model_{}.pth".format(total_test_step))

#writer.close()
```

