"""RepGhost with re-parameterized cheap operations."""

from typing import List

import torch
import torch.nn as nn

from ..base_model import BaseModel


class RepGhostModule(nn.Module):
    """Ghost module using parallel 3x3 and 1x1 branches (train-time multi-branch)."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2) -> None:
        super().__init__()
        init_channels = max(out_channels // ratio, 1)
        new_channels = init_channels * (ratio - 1)

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap_3x3 = nn.Conv2d(init_channels, new_channels, kernel_size=3, padding=1, groups=init_channels, bias=False)
        self.cheap_1x1 = nn.Conv2d(init_channels, new_channels, kernel_size=1, groups=init_channels, bias=False)
        self.bn = nn.BatchNorm2d(new_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        cheap = self.cheap_3x3(x1) + self.cheap_1x1(x1)
        cheap = self.act(self.bn(cheap))
        return torch.cat([x1, cheap], dim=1)


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck with optional stride."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int, stride: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [RepGhostModule(in_channels, hidden)]
        if stride > 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
                    nn.BatchNorm2d(hidden),
                )
            )
        layers.append(RepGhostModule(hidden, out_channels))
        self.block = nn.Sequential(*layers)

        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class RepGhost(BaseModel):
    """RepGhost network for MNIST."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        cfg = [(16, 16, 1), (48, 24, 2), (72, 32, 2), (120, 64, 2), (200, 96, 1)]

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        layers: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(1, ch(16), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch(16)),
                nn.ReLU(inplace=True),
            )
        ]
        in_ch = ch(16)
        for exp, out_c, stride in cfg:
            layers.append(RepGhostBottleneck(in_ch, ch(exp), ch(out_c), stride))
            in_ch = ch(out_c)

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.view(x.size(0), -1))
