import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 314)
        self.linear3 = nn.Linear(314, 128)
        self.linear4 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        data = self.linear1(data)
        data = self.relu(data)
        data = self.linear2(data)
        data = self.relu(data)
        data = self.linear3(data)
        data = self.relu(data)
        data = self.linear4(data)
        data = self.sigmoid(data)
        return data

def main():
    torch.manual_seed(100)
    data = torch.randn(4, 768)
    # 量化前推理
    model = Model()
    output1 = model(data)
    print(output1)
    # 量化模型
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # 量化后推理
    output2 = quantized_model(data)
    print(output2)
    # 查看模型
    print(model)
    print(quantized_model)
    # 模型保存
    torch.save(model, "./model")
    torch.save(quantized_model, "./quantized_model")

if __name__ == "__main__":
    main()

