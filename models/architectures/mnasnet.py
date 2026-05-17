"""MNASNet-style depthwise-separable backbone for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


def _make_divisible(value: int, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class MNASBlock(nn.Module):
    """Depthwise separable block with optional residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
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
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            return x + out
        return out


class MNASNet(BaseModel):
    """Lightweight MNASNet-A0 style macro for 28x28 grayscale."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        # (kernel, expand, out_channels, stride, repeats)
        cfg: List[Tuple[int, int, int, int, int]] = [
            (3, 1, 16, 1, 1),
            (3, 6, 24, 2, 2),
            (5, 6, 40, 2, 3),
            (3, 6, 80, 2, 2),
            (3, 6, 96, 1, 2),
            (5, 6, 192, 2, 3),
            (5, 6, 320, 1, 1),
        ]

        input_ch = _make_divisible(int(32 * width_mult))
        layers: List[nn.Module] = [
            nn.Conv2d(1, input_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU6(inplace=True),
        ]
        in_ch = input_ch
        for k, exp, out_ch, stride, repeats in cfg:
            out_ch = _make_divisible(int(out_ch * width_mult))
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(MNASBlock(in_ch, out_ch, k, s, exp))
                in_ch = out_ch

        last_ch = _make_divisible(int(1280 * width_mult))
        layers.extend(
            [
                nn.Conv2d(in_ch, last_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(last_ch),
                nn.ReLU6(inplace=True),
            ]
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
