"""Network in Network (NiN) architecture."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class MLPConv(nn.Module):
    """1x1 convolutions acting as per-pixel MLP (NiN block)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NiN(BaseModel):
    """Network in Network for MNIST classification."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: Optional[List[int]] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        channels = channels or [96, 256, 384]

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0] // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            MLPConv(channels[0] // 2, channels[0]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = nn.Sequential(
            MLPConv(channels[0], channels[1]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            MLPConv(channels[1], channels[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
