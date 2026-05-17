"""CSPNet with cross-stage partial connections."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class CSPBlock(nn.Module):
    """Split features, process one branch densely, then merge."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        blocks = []
        ch = mid
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, mid, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.conv2 = nn.Conv2d(mid * 2, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        y2 = self.blocks(y1)
        out = torch.cat([y1, y2], dim=1)
        return F.leaky_relu(self.bn2(self.conv2(out)), 0.1, inplace=True)


class CSPStage(nn.Module):
    """CSP stage with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1) -> None:
        super().__init__()
        if stride > 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
            in_channels = out_channels
        else:
            self.down = nn.Identity()
        self.csp = CSPBlock(in_channels, out_channels, num_blocks=num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.csp(x)


class CSPNet(BaseModel):
    """Lightweight CSPNet classifier for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: Optional[List[int]] = None,
        blocks: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        blocks = blocks or [2, 2, 2, 2]

        layers: List[nn.Module] = [
            nn.Conv2d(1, channels[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        in_ch = channels[0] // 2
        for idx, (out_ch, num_blocks) in enumerate(zip(channels, blocks)):
            stride = 2 if idx > 0 else 1
            layers.append(CSPStage(in_ch, out_ch, num_blocks, stride=stride))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
