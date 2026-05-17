"""Dual Path Network (DPN) with parallel residual and dense paths."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class DPNBlock(nn.Module):
    """Dual-path block: residual stream + dense feature reuse."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dense_channels: int = 32,
    ) -> None:
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dense = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, dense_channels, kernel_size=1, bias=False),
        )
        self.out_channels = out_channels + dense_channels

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_feat = self.dense(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if dense_feat.shape[-2:] != out.shape[-2:]:
            dense_feat = F.adaptive_avg_pool2d(dense_feat, out.shape[-2:])
        out = torch.cat([out, dense_feat], dim=1)
        if not isinstance(self.shortcut, nn.Sequential) or len(self.shortcut) == 0:
            return F.relu(out, inplace=True)
        return F.relu(out + self.shortcut(x), inplace=True)


class DPN(BaseModel):
    """Lightweight DPN for 28x28 MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        dense_channels: int = 32,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, dense_channels=dense_channels)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, dense_channels=dense_channels)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, dense_channels=dense_channels)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, dense_channels=dense_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, dense_channels: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(DPNBlock(self.in_channels, out_channels, s, dense_channels=dense_channels))
            self.in_channels = layers[-1].out_channels
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
