import torch 
import torch.nn as nn

class ImageClassificationModel(nn.Module):
    def __init__(self):
        super(ImageClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, stride=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(2704, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
 
    def forward(self, data):
        data = torch.relu(self.conv1(data))
        data = self.pool1(data)
        data = torch.relu(self.conv2(data))
        data = self.pool2(data)
        data = data.reshape(data.size(0), -1) # (N, C, H, W) -> (N, -1)
        data = torch.relu(self.linear1(data))
        data = torch.relu(self.linear2(data))
        output = self.linear3(data)
        print(output.shape)
        return torch.sigmoid(output)
    

def export():
    model = ImageClassificationModel()
    for name, module in model.named_modules():
        print(f"Name: {name}, Type: {type(module).__name__}")
 
    # 执行一次inference, 确保模型搭建正确
    # batch_size, channel, height, weight
    input = torch.randn([4, 3, 60, 60])
    output = model(input)
    print(output) # [4, 10] batch_size, class_number
 
    torch.onnx.export(
        model,                        # PyTorch 模型
        input,                        # 模型输入
        "model.onnx",  # 输出文件路径
        export_params=True,           # 是否导出模型参数
        opset_version=11,             # ONNX 算子集版本
        do_constant_folding=True,     # 是否执行常量折叠优化
        input_names=["input"],        # 输入节点名称
        output_names=["output"],      # 输出节点名称
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 动态轴
    )

if __name__ == "__main__":
    export()
    