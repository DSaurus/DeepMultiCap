import torch
import torch.nn as nn

class IdentityBlock(nn.Module):
    def __init__(self, channels, filters):
        super(IdentityBlock, self).__init__()
        self.channels = channels
        self.filters = filters
        self.net = nn.Sequential(
            nn.Conv3d(channels, filters[0], 1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(True),
            nn.Conv3d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm3d(filters[1]),
            nn.ReLU(True),
            nn.Conv3d(filters[1], filters[2], 1),
            nn.BatchNorm3d(filters[2])
        )
        self.channel_net = nn.Conv3d(channels, filters[2], 1, 1, 0)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.net(x)
        if self.channels != self.filters[2]:
            y = self.channel_net(x) + y
        else:
            y = x + y
        y = self.relu(y)

        return y


class ConvBlock(nn.Module):
    def __init__(self, channels, filters, stride=2):
        super(ConvBlock, self).__init__()
        self.channels = channels
        self.filters = filters
        self.net = nn.Sequential(
            nn.Conv3d(channels, filters[0], 1, stride=stride),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(True),
            nn.Conv3d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm3d(filters[1]),
            nn.ReLU(True),
            nn.Conv3d(filters[1], filters[2], 1),
            nn.BatchNorm3d(filters[2])
        )
        self.downsample =  nn.Sequential(
            nn.Conv3d(channels, filters[2], 1, stride=stride),
            nn.BatchNorm3d(filters[2])
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.net(x)
        x = self.downsample(x)
        y = x + y
        y = self.relu(y)

        return y

class ResNet3d(nn.Module):
    def __init__(self):
        super(ResNet3d, self).__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(1, [16, 16, 32]),
            IdentityBlock(32, [16, 16, 32])
        )
        self.stage_d = nn.Sequential(
            ConvBlock(32, [16, 16, 32]),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 1)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(32, [16, 16, 64]),
            IdentityBlock(64, [32, 32, 64]),
            IdentityBlock(64, [32, 32, 64])
        )
        # self.stage3 = nn.Sequential(
        #     ConvBlock(32, [16, 16, 64]),
        #     IdentityBlock(64, [32, 32, 64]),
        #     IdentityBlock(64, [32, 32, 64]),
        #     IdentityBlock(64, [32, 32, 64])
        # )
    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        # x3 = self.stage3(x2)
        return x2