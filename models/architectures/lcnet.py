"""PP-LCNet style lightweight CNN for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class HSigmoid(nn.Module):
    """Hard sigmoid activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """Hard swish activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * HSigmoid()(x)


class DepthwiseSeparable(nn.Module):
    """Depthwise separable conv with optional SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_se: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        activation = nn.ReLU(inplace=True) if act == "relu" else HSwish()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            activation,
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        ]
        self.block = nn.Sequential(*layers)
        self.se = None
        if use_se:
            reduced = max(out_channels // 4, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, reduced, 1),
                HSigmoid(),
                nn.Conv2d(reduced, out_channels, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.se is not None:
            x = x * self.se(x)
        return x


class LCNet(BaseModel):
    """Lightweight LCNet macro-architecture."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        cfg: List[Tuple[int, int, bool, str]] = [
            (16, 1, False, "relu"),
            (32, 2, False, "relu"),
            (64, 1, True, "relu"),
            (64, 2, False, "relu"),
            (128, 1, True, "hswish"),
            (128, 2, True, "hswish"),
            (256, 1, False, "hswish"),
        ]

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        layers: List[nn.Module] = [
            nn.Conv2d(1, ch(16), 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch(16)),
            HSwish(),
        ]
        in_ch = ch(16)
        for out_c, stride, se, act in cfg:
            out_ch = ch(out_c)
            layers.append(DepthwiseSeparable(in_ch, out_ch, stride, use_se=se, act=act))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
