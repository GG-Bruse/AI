import torch
import numpy as np

data = torch.randint(10, [2, 3])
print(data)

# 不修改原数据
new_data = data.add(10) # 等价于 new_data = data + 10
print(new_data)

# 修改原数据
data.add_(10)
print(data)

# 其他函数
print(data.sub(100))
print(data.mul(100))
print(data.div(100))
print(data.neg())



# 阿达玛积:矩阵对应的元素相乘
data1 = torch.tensor([[1, 2], [3, 4]])
data2 = torch.tensor([[5, 6], [7, 8]])
# 方式一
data = torch.mul(data1, data2)
print(data)
# 方式二
data = data1 * data2
print(data)



# 点积运算
# 要求第一个矩阵shape(n, m), 第二个矩阵shape(m, p), 点积结果shape为(n, p)
data1 = torch.tensor([[1, 2], [3, 4], [5, 6]]) # (3,2)
data2 = torch.tensor([[5, 6], [7, 8]]) # (2, 2)

data = data1 @ data2
print(data)

data = torch.mm(data1, data2) # 要求输入矩阵为2维
print(data)

# torch.matmul 对进行点乘运算的两矩阵形状没有限定
# 对于输入都是二维的张量相当于 mm 运算
# 对于输入都是三维的张量相当于 bmm 运算
# 对数输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则
data = torch.matmul(data1, data2)
print(data)
print(torch.matmul(torch.randn(3, 4, 5), torch.randn(5, 4)).shape)
print(torch.matmul(torch.randn(5, 4), torch.randn(3, 4, 5)).shape)

# 批量点积运算
# 第一个维度为 batch_size
# 矩阵的二三维要满足矩阵乘法规则
data1 = torch.randn(3, 4, 5)
data2 = torch.randn(3, 5, 8)
data = torch.bmm(data1, data2)
print(data.shape)