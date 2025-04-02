import torch 
import numpy as np



print("torch.tensor根据指定数据创建张量")
# 创建张量标量
data = torch.tensor(10)
print(data)
# numpy数组
data = np.random.randn(2, 3) # 两行三列
data = torch.tensor(data)
print(data)
# 列表
data = [[10., 20., 30.], [40., 50., 60.]]
data = torch.tensor(data)
print(data)



print("\ntorch.Tensor根据形状创建张量")
# 两行三列的张量
data = torch.Tensor(2, 3)
print(data)
# 若传递列表, 则创建指定元素的张量
data = torch.Tensor([100.])
print(data)
data = torch.Tensor([200, 300.])
print(data)



print("\n创建指定类型的张量")
data = torch.IntTensor(2, 3) # int32
print(data)
data = torch.ShortTensor(1, 3) # int16
print(data)
data = torch.LongTensor(1, 2) # int64
print(data)
data = torch.FloatTensor(1, 2) # float32
print(data)
data = torch.DoubleTensor(1, 2) # float64
print(data)



print("\n创建线性张量")
data = torch.arange(0, 10, 2)
print(data) # [start, end, step)
data = torch.linspace(0, 9, 10)
print(data) # [start,end, numbers]



print("\n创建随机张量")
