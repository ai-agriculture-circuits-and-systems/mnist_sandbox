"""HarDNet with harmonically dense connections between layers."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class HarDBlock(nn.Module):
    """Concatenate outputs from a chain of convolutions (dense connectivity)."""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True),
                )
            )
            ch = growth_rate
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    """Compress and optionally downsample feature maps."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if stride > 1:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HarDNet(BaseModel):
    """Compact HarDNet macro-architecture for MNIST."""

    def __init__(self, num_classes: int = 10, growth_rate: int = 16) -> None:
        super().__init__()
        cfg: List[Tuple[int, int]] = [(4, 1), (6, 2), (8, 2), (10, 2)]

        layers: List[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        in_ch = 32
        for num_layers, stride in cfg:
            block = HarDBlock(in_ch, growth_rate, num_layers)
            layers.append(block)
            out_ch = block.out_channels
            layers.append(Transition(out_ch, out_ch // 2, stride=stride))
            in_ch = out_ch // 2

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
