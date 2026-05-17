"""Visual Attention Network (VAN) with large-kernel attention."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class LargeKernelAttention(nn.Module):
    """Spatial attention using decomposed large kernels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=channels,
            dilation=3,
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class VANBlock(nn.Module):
    """Inverted bottleneck block with large-kernel attention."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            LargeKernelAttention(hidden),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VAN(BaseModel):
    """VAN-style hierarchical backbone for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        dims = dims or [32, 64, 128, 256]
        depths = depths or [2, 2, 4, 2]

        layers: List[nn.Module] = [
            nn.Conv2d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        ]
        in_ch = dims[0]
        for stage, (dim, depth) in enumerate(zip(dims, depths)):
            if stage > 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, dim, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(dim),
                        nn.GELU(),
                    )
                )
                in_ch = dim
            for _ in range(depth):
                layers.append(VANBlock(in_ch))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
