"""ResNet with Coordinate Attention (encodes position along H and W)."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class CoordAttention(nn.Module):
    """Coordinate attention factorizes channel attention across spatial axes."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch, channels, height, width = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = F.relu(self.bn1(self.conv1(y)), inplace=True)
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class CoordBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.coord = CoordAttention(out_channels, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.coord(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class CoordResNet(BaseModel):
    """ResNet-18 style backbone with coordinate attention."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        reduction: int = 32,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, reduction: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(CoordBasicBlock(self.in_channels, out_channels, s, reduction=reduction))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
