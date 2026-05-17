"""SKNet with selective kernel attention over multiple conv branches."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class SKConv(nn.Module):
    """Selective kernel convolution (3x3 and 5x5 branches)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernels: Tuple[int, ...] = (3, 5),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=stride,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernels
            ]
        )
        mid = max(out_channels // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, mid, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Linear(mid, out_channels * len(kernels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        u = sum(feats)
        batch, channels, _, _ = u.size()
        s = F.adaptive_avg_pool2d(u, 1).view(batch, channels)
        z = self.fc(s)
        attn = self.fc_out(z).view(batch, len(self.branches), channels, 1, 1)
        attn = F.softmax(attn, dim=1)
        out = sum(attn[:, i] * feats[i] for i in range(len(feats)))
        return out


class SKBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        mid = max(out_channels // 2, 32)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.sk = SKConv(mid, out_channels, stride=stride, reduction=reduction)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.sk(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class SKNet(BaseModel):
    """SKNet-18 style backbone for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, reduction: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SKBlock(self.in_channels, out_channels, s, reduction=reduction))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
