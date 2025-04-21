import torch
import torch.nn as nn

# 定义模型
model = nn.Linear(10, 1)
# 定义损失函数，包括L2正则化项
criterion = nn.MSELoss()

# 计算损失
input = torch.randn(1, 10)
target = torch.randn(1, 1)
output = model(input)
loss = criterion(output, target)
# 打印损失
print(loss)