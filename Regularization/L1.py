import torch
import torch.nn as nn

# 构建模型和损失函数(L1)
model = nn.Linear(10, 1)
criterion = nn.L1Loss()

# 输入与目标数据
torch.manual_seed(100)
input = torch.randn([32, 10])
target = torch.randn([32, 1])
print(input)
print(target)

# 推理
output = model(input)
print(output)

# 计算损失
loss = criterion(output, target)
print(loss)
