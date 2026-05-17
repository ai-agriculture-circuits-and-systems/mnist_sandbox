"""ResNeXt with grouped convolutions (cardinality splits)."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class ResNeXtBlock(nn.Module):
    """Bottleneck block with grouped 3x3 conv (ResNeXt style)."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        cardinality: int = 32,
        width_per_group: int = 4,
    ) -> None:
        super().__init__()
        width = cardinality * width_per_group
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNeXt(BaseModel):
    """ResNeXt-18 style backbone adapted for 28x28 MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        cardinality: int = 32,
        width_per_group: int = 4,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, cardinality=cardinality, width_per_group=width_per_group)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        cardinality: int,
        width_per_group: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                ResNeXtBlock(
                    self.in_channels,
                    out_channels,
                    s,
                    cardinality=cardinality,
                    width_per_group=width_per_group,
                )
            )
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
