"""GhostNet-style efficient CNN with cheap linear operations."""

from typing import List

import torch
import torch.nn as nn

from ..base_model import BaseModel


class GhostModule(nn.Module):
    """Generate feature maps from cheap linear transforms of intrinsic features."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2) -> None:
        super().__init__()
        init_channels = max(out_channels // ratio, 1)
        new_channels = init_channels * (ratio - 1)

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(
                init_channels, new_channels, kernel_size=3, padding=1,
                groups=init_channels, bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = init_channels + new_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        return torch.cat([x1, self.cheap(x1)], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost bottleneck block."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [GhostModule(in_channels, hidden_channels)]
        if stride > 1:
            layers.extend([
                nn.Conv2d(
                    hidden_channels, hidden_channels, kernel_size=3, stride=stride,
                    padding=1, groups=hidden_channels, bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
            ])
        layers.append(GhostModule(hidden_channels, out_channels))
        self.block = nn.Sequential(*layers)

        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class GhostNet(BaseModel):
    """GhostNet for MNIST classification."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        cfg = [(16, 16, 1), (48, 24, 2), (72, 32, 2), (120, 64, 2), (200, 96, 1)]

        layers: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(1, max(int(16 * width_mult), 8), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(max(int(16 * width_mult), 8)),
                nn.ReLU(inplace=True),
            )
        ]
        in_channels = max(int(16 * width_mult), 8)
        for exp, out_c, stride in cfg:
            exp = max(int(exp * width_mult), 8)
            out_c = max(int(out_c * width_mult), 8)
            layers.append(GhostBottleneck(in_channels, exp, out_c, stride))
            in_channels = out_c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.view(x.size(0), -1))
