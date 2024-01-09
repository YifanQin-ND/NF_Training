import torch
import torch.nn as nn
import torch.nn.functional as F
from ..src.generate_noise import generate_noise
from ..config import Config


def inf_with_noise(data, weight, noise, bias):
    return F.linear(data, weight + generate_noise(weight, noise), bias)


class IRS_Block(nn.Module):
    def __init__(self, in_planes, out_planes, num_classes, size, noise_block):
        super().__init__()
        self.size = size
        self.avgpool = nn.AdaptiveAvgPool2d(self.size)
        self.noise_block = noise_block
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=in_planes, out_features=out_planes).to(Config.DEVICE)
        self.fc2 = nn.Linear(in_features=out_planes, out_features=num_classes).to(Config.DEVICE)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        identity = x
        # adp_pool to size(adp, adp)
        out = self.avgpool(x)
        out = torch.flatten(out, 1)
        out = inf_with_noise(out, self.fc1.weight, self.noise_block, self.fc1.bias)
        out = self.relu(out)
        out = inf_with_noise(out, self.fc2.weight, self.noise_block, self.fc2.bias)
        return identity, out
