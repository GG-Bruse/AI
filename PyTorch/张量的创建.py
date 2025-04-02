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
# 创建两行三列随机张量
data = torch.randn(2, 3)
print(data)
print("随机数种子:", torch.initial_seed())
# 设置随机数种子
torch.manual_seed(100)
data = torch.randn(2, 3)
print(data)
print("随机数种子:", torch.initial_seed())



print("创建0、1、指定值的张量")
# 创建指定形状的全0张量
data = torch.zeros(2, 3)
print(data)
# 根据其他张量的形状创建全0张量
data = torch.zeros_like(data)
print(data)
# 创建指定形状的全1张量
data = torch.ones(2, 3)
print(data)
# 根据其他张量的形状创建全1张量
data = torch.ones_like(data)
print(data)
# 创建指定形状的指定值张量
data = torch.full([2, 3], 10)
print(data)
# 根据其他张量的形状创建指定值的张量
data = torch.full_like(data, 20)
print(data)



print("\n张量元素类型转换")
data = torch.full([2, 3], 10)
print(data.dtype)
# 1. 转换为其他类型
data = data.type(torch.DoubleTensor)
print(data.dtype)
data = data.type(torch.ShortTensor)
print(data.dtype)
data = data.type(torch.IntTensor)
print(data.dtype)
data = data.type(torch.LongTensor)
print(data.dtype)
data = data.type(torch.FloatTensor)
print(data.dtype)
data = data.type(dtype=torch.float16)
print(data.dtype)
# 2. 转换为其他类型
data = data.double()
print(data.dtype)
data = data.half()
print(data.dtype)
data = data.short()
print(data.dtype)
data = data.int()
print(data.dtype)
data = data.long()
print(data.dtype)
data = data.float()
print(data.dtype)