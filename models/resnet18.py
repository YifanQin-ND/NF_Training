from typing import Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .irs_block import IRS_Block
from ..src.generate_noise import generate_noise


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)


class BasicBlock(nn.Module):
    def __init__(
            self,
            noise_backbone,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
    ):
        super().__init__()
        self.noise_backbone = noise_backbone
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.conv1 = conv3x3(self.in_planes, self.out_planes, stride=self.stride, padding=1)
        self.conv2 = conv3x3(self.out_planes, self.out_planes, stride=1, padding=1)
        self.unit_conv = conv1x1(self.in_planes, self.out_planes, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(self.out_planes)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU()
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
        out = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=self.stride, padding=1)
        out = self.bn1(out)
        out = self.relu(out)
        out = inf_with_noise(out, self.conv2.weight, self.noise_backbone, stride=1, padding=1)
        out = self.bn2(out)
        if self.in_planes != self.out_planes or self.stride != 1:
            identity = self.bn3(inf_with_noise(x, self.unit_conv.weight, self.noise_backbone, stride=self.stride, padding=0))
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            block: Type[BasicBlock],
            layers: List[int],
            noise_backbone
    ):
        super().__init__()
        self.noise_backbone = noise_backbone  # noise var for backbone
        self.conv1 = conv3x3(in_channels, 64, stride=1, padding=1)  # first conv layer
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=num_classes)
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

    def _make_layer(
            self,
            block: Type[BasicBlock],
            in_planes: int,
            out_planes: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        layers = []
        layers.append(block(self.noise_backbone, in_planes, out_planes, stride=stride))
        for _ in range(1, blocks):
            layers.append(block(self.noise_backbone, out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)

        return x


class IRS_ResNet(ResNet):
    def __init__(self, in_channels, num_classes, block: Type[BasicBlock], layers: List[int], noise_backbone, noise_block):
        super().__init__(in_channels, num_classes, block, layers, noise_backbone)
        self.noise_block = noise_block  # noise var for IRS block
        self.irs_block1 = IRS_Block(2304, 512, num_classes, 6, self.noise_block)
        self.irs_block2 = IRS_Block(2304, 512, num_classes, 6, self.noise_block)
        self.irs_block3 = IRS_Block(2048, 512, num_classes, 4, self.noise_block)
        self.irs_block4 = IRS_Block(2304, 512, num_classes, 3, self.noise_block)

    def forward(self, x):
        x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x, out1 = self.irs_block1(x)
        x = self.layer1(x)
        x, out2 = self.irs_block2(x)
        x = self.layer2(x)
        x, out3 = self.irs_block3(x)
        x = self.layer3(x)
        x, out4 = self.irs_block4(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)

        return out1, out2, out3, out4, x


class BasicBlock_test(nn.Module):
    def __init__(
            self,
            noise_backbone,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
    ):
        super().__init__()
        self.conv_noise = [None] * 2
        self.unit_conv_noise = [None]
        self.noise_backbone = noise_backbone
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = conv3x3(self.in_planes, self.out_planes, stride=self.stride, padding=1)
        self.conv2 = conv3x3(self.out_planes, self.out_planes, stride=1, padding=1)
        self.unit_conv = conv1x1(self.in_planes, self.out_planes, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(self.out_planes)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU()

    def epoch_noise(self):
        for i, module in enumerate([self.conv1, self.conv2]):
            self.conv_noise[i] = generate_noise(module.weight, self.noise_backbone)
        self.unit_conv_noise = generate_noise(self.unit_conv.weight, self.noise_backbone)

    def forward(self, x):
        identity = x
        out = F.conv2d(x, self.conv1.weight + self.conv_noise[0], stride=self.stride, padding=1)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.conv2d(out, self.conv2.weight + self.conv_noise[1], padding=1)
        out = self.bn2(out)
        if self.in_planes != self.out_planes or self.stride != 1:
            identity = self.bn3(F.conv2d(x, self.unit_conv.weight + self.unit_conv_noise, stride=self.stride, padding=0))
        out += identity
        out = self.relu(out)
        return out


class ResNet_test(nn.Module):
    def __init__(self, in_channels, num_classes, block: Type[BasicBlock_test], layers: List[int], noise_backbone):
        super().__init__()
        self.conv1_noise = None
        self.fc1_noise = None
        self.noise_backbone = noise_backbone
        self.conv1 = conv3x3(in_channels, 64, stride=1, padding=1)  # first conv layer
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=num_classes)

    def epoch_noise(self):
        for i in range(1, 9):
            block = getattr(self, f'bb{i}')
            block.epoch_noise()
        self.conv1_noise = generate_noise(self.conv1.weight, self.noise_backbone)
        self.fc1_noise = generate_noise(self.fc1.weight, self.noise_backbone)

    def _make_layer(
            self,
            block: Type[BasicBlock_test],
            in_planes: int,
            out_planes: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        layers = []
        layers.append(block(self.noise_backbone, in_planes, out_planes, stride=stride))
        for _ in range(1, blocks):
            layers.append(block(self.noise_backbone, out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.conv2d(x, self.conv1.weight + self.conv1_noise, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + self.fc1_noise, self.fc1.bias)
        return x


def resnet18(in_channels, num_classes, noise_backbone):
    return ResNet(in_channels, num_classes, BasicBlock, [2, 2, 2, 2], noise_backbone)


def resnet18_irs(in_channels, num_classes, noise_backbone, noise_block):
    return IRS_ResNet(in_channels, num_classes, BasicBlock, [2, 2, 2, 2], noise_backbone, noise_block)


def resnet18_test(in_channels, num_classes, noise_backbone):
    return ResNet_test(in_channels, num_classes, BasicBlock_test, [2, 2, 2, 2], noise_backbone)
