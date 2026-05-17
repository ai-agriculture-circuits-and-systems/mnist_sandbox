"""MobileNetV3-Small style network for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


def _make_divisible(value: float, divisor: int = 8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class HSwish(nn.Module):
    """Hard-swish activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class HSigmoid(nn.Module):
    """Hard-sigmoid for SE gating."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-excitation for inverted residuals."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = _make_divisible(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.act = HSigmoid()
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool,
        activation: str,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(in_channels * expand_ratio, 8)
        self.use_res = stride == 1 and in_channels == out_channels
        act = nn.ReLU(inplace=True) if activation == "relu" else HSwish()

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), act])
        layers.extend(
            [
                nn.Conv2d(hidden, hidden, kernel_size, stride, kernel_size // 2, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                act,
            ]
        )
        if use_se:
            layers.append(SEModule(hidden))
        layers.extend([nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            return x + out
        return out


class MobileNetV3(BaseModel):
    """MobileNetV3-Small adapted for 1-channel 28x28 input."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        base_channels = _make_divisible(16 * width_mult, 8)

        # (kernel, exp, out, SE, act, stride)
        cfg: List[Tuple[int, int, int, bool, str, int]] = [
            (3, 1, 16, True, "relu", 1),
            (3, 4, 24, False, "relu", 2),
            (3, 3, 24, False, "relu", 1),
            (5, 3, 40, True, "relu", 2),
            (5, 3, 40, True, "relu", 1),
            (5, 3, 48, True, "relu", 1),
            (5, 6, 96, True, "hswish", 2),
            (5, 6, 96, True, "hswish", 1),
        ]

        layers: List[nn.Module] = [
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            HSwish(),
        ]
        in_ch = base_channels
        for k, exp, out_ch, se, act, stride in cfg:
            out_ch = _make_divisible(out_ch * width_mult, 8)
            layers.append(InvertedResidual(in_ch, out_ch, k, stride, exp, se, act))
            in_ch = out_ch

        last_ch = _make_divisible(576 * width_mult, 8)
        layers.extend(
            [
                nn.Conv2d(in_ch, last_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(last_ch),
                HSwish(),
            ]
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(last_ch, _make_divisible(1024 * width_mult, 8)),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(_make_divisible(1024 * width_mult, 8), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
