import torchvision
import torch
import torch.nn as nn


class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features=100,out_features=300),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=300,out_features=500),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=500,out_features=1000),
            nn.ReLU()

        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=1000,out_features=500),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=300, out_features=50),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=50, out_features=3),
        )

    def forward(self,x):
        x = x
        x = self.input(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.output(x)
        return x

