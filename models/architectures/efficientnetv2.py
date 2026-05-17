"""EfficientNetV2-style MBConv blocks for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn

from ..base_model import BaseModel


class SEModule(nn.Module):
  """Squeeze-and-excitation."""

  def __init__(self, channels: int) -> None:
    super().__init__()
    reduced = max(channels // 4, 4)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channels, reduced),
      nn.SiLU(inplace=True),
      nn.Linear(reduced, channels),
      nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, channels, _, _ = x.size()
    scale = self.pool(x).view(batch, channels)
    scale = self.fc(scale).view(batch, channels, 1, 1)
    return x * scale


class MBConvV2(nn.Module):
  """MBConv with SiLU and squeeze-excitation."""

  def __init__(self, in_channels: int, out_channels: int, expand: int, kernel: int, stride: int) -> None:
    super().__init__()
    hidden = in_channels * expand
    self.use_res = stride == 1 and in_channels == out_channels
    layers: List[nn.Module] = []
    if expand != 1:
      layers.extend(
        [
          nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
          nn.BatchNorm2d(hidden),
          nn.SiLU(inplace=True),
        ]
      )
    layers.extend(
      [
        nn.Conv2d(hidden, hidden, kernel_size=kernel, stride=stride, padding=kernel // 2, groups=hidden, bias=False),
        nn.BatchNorm2d(hidden),
        nn.SiLU(inplace=True),
      ]
    )
    self.conv = nn.Sequential(*layers)
    self.se = SEModule(hidden)
    self.project = nn.Sequential(
      nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_channels),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.conv(x)
    out = self.se(out)
    out = self.project(out)
    if self.use_res:
      return x + out
    return out


class EfficientNetV2(BaseModel):
  """EfficientNetV2-S style macro for grayscale input."""

  def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
    super().__init__()
    cfg: List[Tuple[int, int, int, int]] = [
      (1, 24, 3, 1),
      (4, 48, 3, 2),
      (4, 64, 3, 2),
      (4, 128, 3, 2),
      (6, 160, 3, 1),
      (6, 256, 3, 2),
    ]

    def ch(c: int) -> int:
      return max(int(c * width_mult), 8)

    layers: List[nn.Module] = [
      nn.Conv2d(1, ch(24), kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(ch(24)),
      nn.SiLU(inplace=True),
    ]
    in_ch = ch(24)
    for expand, out_c, kernel, stride in cfg:
      out_ch = ch(out_c)
      layers.append(MBConvV2(in_ch, out_ch, expand, kernel, stride))
      in_ch = out_ch

    self.features = nn.Sequential(*layers)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(in_ch, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.fc(x)
