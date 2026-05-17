"""Inception-ResNet v2 style blocks for MNIST."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class InceptionResNetBlock(nn.Module):
    """Inception module with residual connection."""

    def __init__(self, in_channels: int, scale: float = 0.17) -> None:
        super().__init__()
        branch_ch = max(in_channels // 4, 16)
        self.branch1 = nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )
        self.conv_linear = nn.Conv2d(branch_ch * 3, in_channels, kernel_size=1, bias=False)
        self.scale = scale
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        out = self.conv_linear(out)
        out = self.bn(x + self.scale * out)
        return self.relu(out)


class InceptionResNet(BaseModel):
    """Inception-ResNet macro model for 28x28 input."""

    def __init__(
        self,
        num_classes: int = 10,
        blocks: Optional[List[int]] = None,
        channels: int = 192,
    ) -> None:
        super().__init__()
        blocks = blocks or [3, 4, 4, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        layers: List[nn.Module] = []
        ch = channels
        for stage_blocks in blocks:
            for _ in range(stage_blocks):
                layers.append(InceptionResNetBlock(ch))
            layers.extend(
                [
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
