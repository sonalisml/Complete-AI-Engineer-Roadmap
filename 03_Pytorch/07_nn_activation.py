import torch
import torch.nn as nn


class MyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 8)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(8, 4)
        self.relu2 = nn.ReLU()

        self.output = nn.Linear(4, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.output(x)

        return x


model = MyNetwork()

print(model)