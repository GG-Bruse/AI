import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(768, 254),
            nn.ReLU(),
            nn.Linear(254, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, data):
        return self.layer(data)
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    
    def compute_l2_loss(self, w):
        return torch.pow(w, 2.).sum()

torch.manual_seed(100)
x = torch.randn([5, 768])
y = torch.randn([5, 10])

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

outputs = model(x)
loss = criterion(outputs, y)

# Elastic Net正则化
l1_weight = 0.3
l2_weight = 0.7
parameters = []
for parameter in model.parameters():
    parameters.append(parameter.view(-1))
l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
loss += l1 + l2

loss.backward()
optimizer.step()
