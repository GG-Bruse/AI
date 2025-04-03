import torch


# 单标量的梯度求导
x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
f = x ** 2 + 10
f.backward() # 自动微分
print("x变量的梯度:", x.grad)


# 单向量梯度的计算
x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)
f1 = x ** 2 + 20
print(f1)
# 注意:
# 由于求导的结果必须是标量
# 而 f 的结果是: tensor([120.,  420.,  920., 1620.])
# 所以, 不能直接自动微分, 需要将结果计算为标量才能进行计算
f2 = f1.mean()  # f2 = 1/4 * x
print(f2)
f2.backward()
print("x变量的梯度:", x.grad)


# 多标量梯度计算
x1 = torch.tensor(10, requires_grad=True, dtype=torch.float64)
x2 = torch.tensor(20, requires_grad=True, dtype=torch.float64)
y = x1 ** 2 + x2 ** 2 + x1 * x2
y = y.sum()
y.backward()
print(x1.grad, x2.grad)


# 多向量梯度计算
x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
x2 = torch.tensor([30, 40], requires_grad=True, dtype=torch.float64)
y = x1 ** 2 + x2 ** 2 + x1 * x2
print(y)
y = y.sum()
print(y)
y.backward()
print(x1.grad, x2.grad)



# 不进行梯度计算
x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
print(x.requires_grad)
 # 第一种方式: 对代码进行装饰
with torch.no_grad():
    y = x ** 2
print(y.requires_grad)
# 第二种方式: 对函数进行装饰
@torch.no_grad()
def my_func(x):
    return x ** 2
print(my_func(x).requires_grad)
# 第三种方式
torch.set_grad_enabled(False)
y = x ** 2
print(y.requires_grad)



# 累计梯度
torch.set_grad_enabled(True)
x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)
for _ in range(3):
    f1 = x ** 2 + 20
    f2 = f1.mean()
    # 默认张量的grad属性会累计历史梯度值，所以需要每次手动清理上次的梯度
    # 注意: 一开始梯度不存在, 需要做判断
    if x.grad is not None:
        x.grad.data.zero_()
    f2.backward()
    print(x.grad)



# 梯度下降优化最优解
x = torch.tensor([10], requires_grad=True, dtype=torch.float64)
for _ in range(10):
    f = x ** 2 # 正向计算
    # 梯度清0
    if x.grad is not None:
        x.grad.data.zero_()
    # 反向传播计算梯度
    f.backward()
    # 更新参数
    x.data = x.data - 0.001 * x.grad
    print('%.10f' % x.data)



# 当对设置 requires_grad=True 的张量使用 numpy 函数进行转换时, 会出现如下报错:
# Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
# 此时, 需先使用 detach 函数将张量进行分离, 再使用 numpy 函数
# 注意: detach 之后会产生一个新的张量, 新的张量作为叶子结点，并且该张量和原来的张量共享数据, 但是分离后的张量不需要计算梯度

x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
# print(x1.numpy())  # 错误
print(x1.detach().numpy())  # 正确

# detach 前后张量共享内存 （注意: x1.data 中 返回的是一个与x1共享底层数据的新张量, id(x1.data) 相当于取这个新张量的地址, 不能通过这个方式来判断是否共享内存）
x2 = x1.detach()
print(x1.data_ptr(), x2.data_ptr())
x2[0] = 0
print(x1.data_ptr(), x2.data_ptr())
print(x1.data)
print(x2.data)
x2.data = torch.tensor([100, 200])
print(x1.data_ptr(), x2.data_ptr())
print(x1.data)
print(x2.data)

# x2 不会自动计算梯度: False
print(x2.requires_grad)
