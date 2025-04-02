import torch
import numpy as np

print("Tensor转numpy数组")

data_tensor = torch.tensor([2, 3, 4])
print(data_tensor)
# 将张量转换为numpy数组
data_numpy = data_tensor.numpy()
print(type(data_tensor))
print(type(data_numpy))
# 注意: data_tensor和data_numpy共享内存, 修改其中一个另外一个也会被修改
data_numpy[0] = 100
print(data_tensor)
print(data_numpy)

# 对象拷贝避免共享内存
data_numpy = data_tensor.numpy().copy()
print(type(data_tensor))
print(type(data_numpy))
data_numpy[0] = 1000
print(data_tensor)
print(data_numpy)



print("\nnumpy数组转Tensor")
data_numpy = np.array([2, 3, 4, 5])
data_tensor = torch.from_numpy(data_numpy)
# numpy数组和Tensor共享内存
data_tensor[0] = 100
print(data_tensor)
print(data_numpy)

data_tensor = torch.tensor(data_numpy)
# 此时不共享内存
data_tensor[0] = 1000
print(data_tensor)
print(data_numpy)



print("\n标量张量与数字转换")
data = torch.tensor(10)
print(data.item())
data = torch.tensor([10,])
print(data.item())
