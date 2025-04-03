import torch
import torch.nn as nn
import torch.optim as optim
import pickle

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        output = self.linear2(inputs)
        return output

# 直接序列化模型
def test01():
    model = Model(10, 7)
    # 第三个参数: 使用的模块
    # 第四个参数: 存储的协议
    torch.save(model, "./model/model.pth", pickle_module=pickle, pickle_protocol=2)

# 加载模型
def test02():
    # 第三个参数: 加载的模块
    model = torch.load('model/model.pth', map_location='cpu', pickle_module=pickle)
    data = torch.FloatTensor(5, 10)
    result = model(data)
    print(result)

# 存储模型的网络参数
def test03():
    model = Model(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 定义存储参数
    save_params = {
        'init_params': {
            'input_size': 128,
            'output_size': 10
        },
        'acc_score': 0.98,
        'avg_loss': 0.86,
        'iter_numbers': 100,
        'optim_params': optimizer.state_dict(),
        'model_params': model.state_dict()
    }
    torch.save(save_params, 'model/model_params.pth')

def test04():
    # 加载模型参数
    model_params = torch.load('model/model_params.pth', weights_only=True)
    # 初始化模型
    model = Model(model_params['init_params']['input_size'], model_params['init_params']['output_size'])
    # 初始化优化器
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(model_params['optim_params'])
    # 显示其他参数
    print('迭代次数:', model_params['iter_numbers'])
    print('准确率:', model_params['acc_score'])
    print('平均损失:', model_params['avg_loss'])
    data = torch.FloatTensor(5, 128)
    result = model(data)
    print(result)


if __name__ == "__main__":
    # test01()
    # test02()
    test03()
    test04()

