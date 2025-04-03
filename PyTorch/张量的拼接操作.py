import torch
import numpy

# 将张量按照指定维度拼接
torch.manual_seed(100)
data1 = torch.randint(0, 10, [3, 5, 4])
data2 = torch.randint(0, 10, [3, 5, 4])
print(data1)
print(data2)
new_data = torch.cat([data1, data2], dim=0) # 按照0维度拼接
print(new_data.shape)
new_data = torch.cat([data1, data2], dim=1) # 按照1维度拼接
print(new_data.shape)
new_data = torch.cat([data1, data2], dim=2) # 按照2维度拼接
print(new_data.shape)

# 将两个张量按照指定维度叠加起来
data1= torch.randint(0, 10, [2, 3])
data2= torch.randint(0, 10, [2, 3])
print(data1)
print(data2)
new_data = torch.stack([data1, data2], dim=0)
print(new_data)
print(new_data.shape)
new_data = torch.stack([data1, data2], dim=1)
print(new_data)
print(new_data.shape)
new_data = torch.stack([data1, data2], dim=2)
print(new_data)
print(new_data.shape)