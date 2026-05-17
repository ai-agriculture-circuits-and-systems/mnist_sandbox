"""CoAtNet: hybrid convolution and relative self-attention."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class RelativeAttention2d(nn.Module):
    """Lightweight 2D relative attention block."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.reshape(batch, self.num_heads, channels // self.num_heads, -1)
        k = k.reshape(batch, self.num_heads, channels // self.num_heads, -1)
        v = v.reshape(batch, self.num_heads, channels // self.num_heads, -1)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(batch, channels, height, width)
        return x + self.proj(out)


class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand: int = 4) -> None:
        super().__init__()
        hidden = in_channels * expand
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class CoAtNet(BaseModel):
    """CoAtNet-lite for 224x224 grayscale classification."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 224,
        dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        dims = dims or [64, 128, 256, 512]
        depths = depths or [2, 2, 4, 2]

        layers: List[nn.Module] = [
            nn.Conv2d(1, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        ]
        in_ch = dims[0]
        for stage, (dim, depth) in enumerate(zip(dims, depths)):
            if stage > 0:
                layers.append(MBConv(in_ch, dim, stride=2))
                in_ch = dim
            for block_idx in range(depth):
                if block_idx % 2 == 0:
                    layers.append(MBConv(in_ch, in_ch))
                else:
                    layers.append(RelativeAttention2d(in_ch, num_heads=max(in_ch // 32, 1)))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
