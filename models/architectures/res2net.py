"""Res2Net with multi-scale feature fusion inside each block."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class Res2NetBlock(nn.Module):
    """Residual block that splits channels into scale groups."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        scale: int = 4,
        base_width: int = 26,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0))
        self.scale = scale
        self.width = width

        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        convs = []
        bns = []
        for i in range(scale - 1):
            conv_stride = stride if i == 0 else 1
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=conv_stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        spx = torch.split(out, self.width, dim=1)

        outputs = []
        sp = spx[0]
        sp = F.relu(self.bns[0](self.convs[0](sp)), inplace=True)
        outputs.append(sp)
        for i in range(1, self.scale - 1):
            branch = spx[i]
            if branch.shape[-2:] != sp.shape[-2:]:
                branch = F.adaptive_avg_pool2d(branch, sp.shape[-2:])
            sp = sp + branch
            sp = F.relu(self.bns[i](self.convs[i](sp)), inplace=True)
            outputs.append(sp)

        last = spx[self.scale - 1]
        if last.shape[-2:] != sp.shape[-2:]:
            last = F.adaptive_avg_pool2d(last, sp.shape[-2:])
        outputs.append(last)
        h = min(o.size(2) for o in outputs)
        w = min(o.size(3) for o in outputs)
        outputs = [F.adaptive_avg_pool2d(o, (h, w)) for o in outputs]
        out = torch.cat(outputs, dim=1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class Res2Net(BaseModel):
    """Res2Net-18 style backbone for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: Optional[List[int]] = None,
        scale: int = 4,
        base_width: int = 26,
    ) -> None:
        super().__init__()
        num_blocks = num_blocks or [2, 2, 2, 2]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, scale=scale, base_width=base_width)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, scale=scale, base_width=base_width)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, scale=scale, base_width=base_width)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, scale=scale, base_width=base_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        scale: int,
        base_width: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Res2NetBlock(self.in_channels, out_channels, s, scale=scale, base_width=base_width))
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
