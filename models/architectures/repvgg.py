"""RepVGG-style reparameterizable CNN (multi-branch blocks at training time)."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class RepVGGBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 + optional identity branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = (
            nn.BatchNorm2d(in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return F.relu(out, inplace=True)


class RepVGG(BaseModel):
    """Lightweight RepVGG-A0 style network for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        blocks_per_stage: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        blocks_per_stage = blocks_per_stage or [2, 4, 6, 2]
        channels = [int(c * width_mult) for c in [64, 128, 256, 512]]

        layers: List[nn.Module] = [RepVGGBlock(1, channels[0], stride=2)]
        in_ch = channels[0]

        for stage_i, (out_ch, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            for block_i in range(num_blocks):
                if block_i == 0 and stage_i > 0:
                    layers.append(RepVGGBlock(in_ch, out_ch, stride=2))
                    in_ch = out_ch
                else:
                    layers.append(RepVGGBlock(in_ch, in_ch, stride=1))

        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))
