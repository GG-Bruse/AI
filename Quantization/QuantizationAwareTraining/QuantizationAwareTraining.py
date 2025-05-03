import torch
import torch.nn as nn
import torch.ao.quantization as quantization

torch.manual_seed(123)

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
    torch.manual_seed(123)
    train_data = torch.randn(30000, 768)
    torch.manual_seed(123)
    train_label = torch.randn(30000, 64)

    model = Model()

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
    model_prepared = quantization.prepare_qat(model)

    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.01)

    for i in range(10):
        preds = model_prepared(train_data)
        loss = torch.nn.functional.mse_loss(preds, train_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("loss:", i, loss)


if __name__ == "__main__":
    main()