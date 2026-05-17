"""ResNet with Efficient Channel Attention (ECA)."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class ECALayer(nn.Module):
    """1D conv channel attention without dimensionality reduction."""

    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, 1, channels)
        y = self.conv(y).view(batch, channels, 1, 1)
        return x * torch.sigmoid(y)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, k_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.eca = ECALayer(out_channels, k_size=k_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ECAResNet(BaseModel):
    """ResNet-18 style backbone with ECA attention."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        k_size: int = 3,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, k_size=k_size)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, k_size=k_size)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, k_size=k_size)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, k_size=k_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, k_size: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ECABasicBlock(self.in_channels, out_channels, s, k_size=k_size))
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
