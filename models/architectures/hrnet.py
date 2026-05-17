"""HRNet with parallel multi-resolution branches."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class BasicStage(nn.Module):
    """Stack of 3x3 conv blocks."""

    def __init__(self, channels: int, num_blocks: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HRNet(BaseModel):
    """Two-branch HRNet-lite for MNIST."""

    def __init__(self, num_classes: int = 10, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.branch1 = BasicStage(c1, num_blocks=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            BasicStage(c2, num_blocks=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c1 + c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b2_up = F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.head(torch.cat([b1, b2_up], dim=1))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
