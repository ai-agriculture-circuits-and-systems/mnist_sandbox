"""Wide Residual Network (WideResNet)."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class WideBasicBlock(nn.Module):
    """Wide ResNet basic block with dropout."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(BaseModel):
    """Wide ResNet for MNIST (depth=28, widen_factor=10 style, configurable)."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("depth must satisfy (depth - 4) % 6 == 0")
        n = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(channels[0], channels[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[1], channels[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[2], channels[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int, dropout: float
    ) -> nn.Sequential:
        layers = [WideBasicBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
