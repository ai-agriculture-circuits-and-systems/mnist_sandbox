"""ShuffleNet V2-style efficient CNN."""

from typing import List, Optional

import torch
import torch.nn as nn

from ..base_model import BaseModel


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels across groups."""
    batch_size, channels, height, width = x.size()
    x = x.view(batch_size, groups, channels // groups, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)


class ShuffleUnit(nn.Module):
    """ShuffleNet V2 unit block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        self.stride = stride

        if stride == 1:
            self.branch1 = nn.Sequential()
            branch2_in = in_channels // 2
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            branch2_in = mid_channels

        self.branch2 = nn.Sequential(
            nn.Conv2d(branch2_in, branch2_in, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch2_in, branch2_in, kernel_size=3, stride=stride, padding=1, groups=branch2_in),
            nn.BatchNorm2d(branch2_in),
            nn.Conv2d(branch2_in, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1 = self.branch1(x)
            x2 = x
        x2 = self.branch2(x2)
        out = torch.cat([x1, x2], dim=1)
        return channel_shuffle(out, groups=2)


class ShuffleNet(BaseModel):
    """ShuffleNet V2 for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        stages: Optional[List[int]] = None,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        stages = stages or [4, 8, 4]

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        layers: List[nn.Module] = []
        in_channels = base_channels
        for i, num_units in enumerate(stages):
            out_channels = base_channels * (2 ** (i + 1))
            for j in range(num_units):
                stride = 2 if j == 0 else 1
                layers.append(ShuffleUnit(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)
