import torch
import torch.nn as nn
import torch.nn.functional as F

from .irs_block import IRS_Block
from ..src.generate_noise import generate_noise


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, noise_backbone):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.noise_backbone = noise_backbone

        self.conv1 = conv3x3(self.in_channels, 64, stride=1, padding=1)
        self.conv2 = conv3x3(64, 128, stride=1, padding=1)
        self.conv3 = conv3x3(128, 256, stride=1, padding=1)
        self.conv4 = conv3x3(256, 256, stride=1, padding=1)
        self.conv5 = conv3x3(256, 512, stride=1, padding=1)
        self.conv6 = conv3x3(512, 512, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=8192, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.num_classes)
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

        x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv2.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = inf_with_noise(x, self.conv3.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn3(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv4.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)

        x = inf_with_noise(x, self.conv5.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn5(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv6.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)
        x = self.relu(x)
        output = F.linear(x, self.fc2.weight + generate_noise(self.fc2.weight, self.noise_backbone), self.fc2.bias)

        return output


class IRS_VGG(VGG):
    def __init__(self, in_channels, num_classes, noise_backbone, noise_block):
        super().__init__(in_channels, num_classes, noise_backbone)
        self.noise_block = noise_block
        self.irs_block1 = IRS_Block(2304, 512, num_classes, 6, self.noise_block)
        self.irs_block2 = IRS_Block(2048, 512, num_classes, 4, self.noise_block)
        self.irs_block3 = IRS_Block(2304, 512, num_classes, 3, self.noise_block)

    def forward(self, x):

        x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)
        x, out1 = self.irs_block1(x)

        x = inf_with_noise(x, self.conv2.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)
        x, out2 = self.irs_block2(x)

        x = inf_with_noise(x, self.conv3.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn3(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv4.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)
        x, out3 = self.irs_block3(x)

        x = inf_with_noise(x, self.conv5.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn5(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv6.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)
        x = self.relu(x)
        output = F.linear(x, self.fc2.weight + generate_noise(self.fc2.weight, self.noise_backbone), self.fc2.bias)

        return out1, out2, out3, output


class VGG_TEST(nn.Module):
    def __init__(self, in_channels, num_classes, noise_backbone):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.noise_backbone = noise_backbone
        self.conv_noise = [None] * 6
        self.fc_noise = [None] * 2

        self.conv1 = conv3x3(self.in_channels, 64, stride=1, padding=1)
        self.conv2 = conv3x3(64, 128, stride=1, padding=1)
        self.conv3 = conv3x3(128, 256, stride=1, padding=1)
        self.conv4 = conv3x3(256, 256, stride=1, padding=1)
        self.conv5 = conv3x3(256, 512, stride=1, padding=1)
        self.conv6 = conv3x3(512, 512, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=8192, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def epoch_noise(self):
        for i, module in enumerate([self.conv1, self.conv2, self.conv3,
                                    self.conv4, self.conv5, self.conv6]):
            self.conv_noise[i] = generate_noise(module.weight, self.noise_backbone)
        for i, module in enumerate([self.fc1, self.fc2]):
            self.fc_noise[i] = generate_noise(module.weight, self.noise_backbone)

    def forward(self, x):

        x = F.conv2d(x, self.conv1.weight + self.conv_noise[0], padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.conv2d(x, self.conv2.weight + self.conv_noise[1], padding=1)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = F.conv2d(x, self.conv3.weight + self.conv_noise[2], padding=1)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.conv2d(x, self.conv4.weight + self.conv_noise[3], padding=1)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)

        x = F.conv2d(x, self.conv5.weight + self.conv_noise[4], padding=1)
        x = self.bn5(x)
        x = self.relu(x)

        x = F.conv2d(x, self.conv6.weight + self.conv_noise[5], padding=1)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + self.fc_noise[0], self.fc1.bias)
        x = self.relu(x)
        output = F.linear(x, self.fc2.weight + self.fc_noise[1], self.fc2.bias)

        return output


def vgg8(in_channels, num_classes, noise_backbone):
    return VGG(in_channels, num_classes, noise_backbone)


def vgg8_irs(in_channels, num_classes, noise_backbone, noise_block):
    return IRS_VGG(in_channels, num_classes, noise_backbone, noise_block)


def vgg8_test(in_channels, num_classes, noise_backbone):
    return VGG_TEST(in_channels, num_classes, noise_backbone)