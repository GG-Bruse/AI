import torch

torch.manual_seed(1000)
data = torch.randint(0, 10, [4, 5])
print(data)

print(data[0]) #行索引
print(data[:, 0]) # 列索引

# 列表索引
print(data[[0, 1], [1, 2]]) # 返回 (0, 1)、(1, 2) 两个位置的元素
print(data[[[0], [1]], [1, 2]]) # 返回 0、1 行的 1、2 列共4个元素

# 范围索引
print(data[:3, :2]) # 前3行的前2列数据
print(data[2:, :2]) # 第2行到最后的前2列数据

# 布尔索引
print(data[data[:, 2] > 5]) # 第三列大于5的行数据
print(data[:, data[1] > 5]) # 第二行大于5的列数据

# 多维索引
data = torch.randint(0, 10, [3, 4, 5])
print(data)
print(data[0, :, :])
print(data[:, 0, :])
print(data[:, :, 0])
