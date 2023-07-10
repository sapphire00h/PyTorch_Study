import torch

outputs=torch.tensor([[0.1,0.2],
                     [0.3,0.4]])
#outputs.argmax(1)输出最大值所在位置，横向输出
#outputs.argmax(0)输出最大值所在位置，纵向输出


preds=outputs.argmax(1)
targets=torch.tensor([0,1])
print(preds==targets)
print((preds==targets).sum())
