"""RegNet-style CNN with parameterized width/depth."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class RegNetBlock(nn.Module):
    """RegNet bottleneck block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, group_width: int) -> None:
        super().__init__()
        mid_channels = max(out_channels // 4, group_width)
        groups = max(mid_channels // group_width, 1)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
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
        return F.relu(out + self.shortcut(x), inplace=True)


class RegNet(BaseModel):
    """RegNet-Y 400MF style configuration for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        widths: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        group_width: int = 8,
    ) -> None:
        super().__init__()
        widths = widths or [32, 64, 128, 256]
        depths = depths or [1, 2, 6, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        layers: List[nn.Module] = []
        in_channels = widths[0]
        for stage_idx, (width, depth) in enumerate(zip(widths, depths)):
            for block_idx in range(depth):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                if block_idx == 0 and stage_idx > 0:
                    in_channels = widths[stage_idx - 1]
                layers.append(RegNetBlock(in_channels, width, stride, group_width))
                in_channels = width

        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)
