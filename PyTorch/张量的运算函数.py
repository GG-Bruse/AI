import torch
import numpy as np

data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
print(data)

# 计算均值（注意:tensor必须为float或者double）
print(data.mean())
print(data.mean(dim=0)) # 按列计算均值
print(data.mean(dim=1)) # 按行计算均值

# 计算总和
print(data.sum())
print(data.sum(dim=0))
print(data.sum(dim=1))

# 计算次方
print(data.pow(2))

# 计算平方根
print(data.sqrt())

# 指数计算, e^n 次方
print(data.exp())

# 对数计算
print(data.log())  # 以 e 为底
print(data.log2())
print(data.log10())
