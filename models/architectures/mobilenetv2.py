"""MobileNetV2 with inverted residual blocks."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class InvertedResidual(nn.Module):
    """Inverted residual block with linear bottlenecks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res:
            return x + out
        return out


class MobileNetV2(BaseModel):
    """MobileNetV2 for MNIST."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        cfg: List[Tuple[int, int, int]] = [
            (1, 16, 1),
            (6, 24, 2),
            (6, 32, 3),
            (6, 64, 4),
            (6, 96, 3),
            (6, 160, 3),
            (6, 320, 1),
        ]

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        layers: List[nn.Module] = [
            nn.Conv2d(1, ch(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU6(inplace=True),
        ]
        in_ch = ch(32)
        for expand, out_c, n in cfg:
            out_ch = ch(out_c)
            for i in range(n):
                stride = 2 if i == 0 and out_c > 32 else 1
                layers.append(InvertedResidual(in_ch, out_ch, stride, expand))
                in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
