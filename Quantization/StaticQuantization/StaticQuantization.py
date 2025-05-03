import torch
import torch.nn as nn
import torch.ao.quantization as quantization

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 314)
        self.linear3 = nn.Linear(314, 128)
        self.linear4 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        data = self.quant(data)
        data = self.linear1(data)
        data = self.linear2(data)
        data = self.dequant(data)
        data = self.relu(data)

        data = self.quant(data)
        data = self.linear3(data)
        data = self.dequant(data)
        data = self.relu(data)

        data = self.quant(data)
        data = self.linear4(data)
        data = self.dequant(data)

        data = self.sigmoid(data)
        return data
    
def main():
    torch.manual_seed(100)
    data = torch.randn(2, 768)
    # 量化前推理
    model = Model()
    output1 = model(data)
    print(output1)
    
    # 量化模型
    weight_observer = quantization.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, quant_min=-128, quant_max=127
    )
    activation_observer = quantization.MinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=255
    )
    qconfig = quantization.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )
    model.qconfig = qconfig
    model_prepared = quantization.prepare(model)
    model_prepared(data)
    model_int8 = quantization.convert(model_prepared)

    # 量化后推理
    output2 = model_int8(data)
    print(output2)
    # 查看模型
    print(model)
    print(model_int8)

    print("int8 model linear1 parameter (int8):\n", torch.int_repr(model_int8.linear1.weight()))
    print("int8 model linear1 parameter:\n", model_int8.linear1.weight())

    # 模型保存
    torch.save(model, "./model")
    torch.save(model_int8, "./model_int8")

if __name__ == "__main__":
    main()