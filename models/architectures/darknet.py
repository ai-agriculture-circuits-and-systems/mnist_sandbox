"""Darknet-style backbone (YOLO family) for classification."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class DarknetBlock(nn.Module):
    """1x1 expand then 3x3 conv with residual shortcut."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class Darknet(BaseModel):
    """Compact Darknet-53 style classifier for MNIST."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        cfg: List[Tuple[int, int]] = [(32, 1), (64, 2), (128, 8), (256, 8), (512, 4)]

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        layers: List[nn.Module] = [
            nn.Conv2d(1, ch(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        in_ch = ch(32)
        for out_c, num_blocks in cfg:
            out_ch = ch(out_c)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            for _ in range(num_blocks):
                layers.append(DarknetBlock(out_ch))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
