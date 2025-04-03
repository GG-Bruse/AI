import torch
import numpy as np



# reshape 函数可以在保证张量数据不变的前提下改变数据的维度，将其转换成指定的形状
torch.manual_seed(100)
data = torch.randn(7, 4)
print(data)
print(data.shape)

new_data = data.reshape(1, 28)
print(new_data)
print(new_data.shape)



# transpose 函数可以实现交换张量形状的指定维度。如:一个张量的形状为(2, 3, 4)可以通过transpose函数把3和4进行交换, 将张量的形状变为(2, 4, 3)
data = torch.randn(2, 3, 4)
print(data)
print(data.shape)
new_data = torch.transpose(data, 1, 2)
print(new_data)
print(new_data.shape)



# permute 函数可以一次交换更多的维度
data = torch.randn(2, 3, 4)
print(data)
print(data.shape)
new_data = torch.permute(data, [1, 2, 0])
print(new_data)
print(new_data.shape)



# view 函数也可以用于修改张量的形状，但是其用法比较局限，只能用于存储在整块内存中的张量
# 在 PyTorch 中，有些张量是由不同的数据块组成的，并没有存储在整块的内存中，view 函数无法对这样的张量进行变形处理
# 如: 一个张量经过了 transpose 或者 permute 函数的处理之后，就无法使用 view 函数进行形状操作
data = torch.tensor([[10, 20, 30], [40, 50, 60]])
print('data shape:', data.size())

new_data = data.view(3, 2)
print(new_data)
print('new_data shape:', new_data.shape)
print('data:', data.is_contiguous()) # 判断张量是否使用整块内存

new_data = torch.transpose(data, 0, 1)
print('new_data:', new_data.is_contiguous())

# 需先使用 contiguous 函数转换为整块内存的张量，再使用 view 函数
print(new_data.contiguous().is_contiguous())
new_data = new_data.contiguous().view(2, 3)
print('new_data shape:', new_data.shape)



# squeeze 函数用删除 shape 为 1 的维度，unsqueeze 在每个维度添加 1, 以增加数据的形状
data = torch.tensor(np.random.randint(0, 10, [1, 3, 1, 5]))
print(data.shape)
# 指定删除指定位置为0的维度，若指定位置维度不是1则不删除
new_data = data.squeeze(0)
print(new_data.shape)
# 删除全部
new_data = data.squeeze()
print(new_data.shape)
# 增加一个维度
new_data = new_data.unsqueeze(1)
print(new_data.shape)
