"""Swin Transformer (tiny) for image classification."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into windows."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """Reverse window partition."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_index", relative.sum(-1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_windows, n, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, n, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[self.relative_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, n, channels)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Swin transformer block with optional shifted windows."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = x.view(batch, height * width, channels)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Downsample by merging 2x2 patches."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, int, int]:
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        height, width = height // 2, width // 2
        return x.view(batch, height * width, -1), height, width


class SwinTiny(BaseModel):
    """Swin-Tiny adapted for grayscale classification."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        patches = img_size // patch_size
        self.patches = (patches, patches)

        self.blocks = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        height, width = patches, patches
        for stage, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage_blocks = nn.ModuleList()
            for block_idx in range(depth):
                shift = 0 if block_idx % 2 == 0 else window_size // 2
                stage_blocks.append(SwinBlock(dim, heads, window_size, shift))
            self.blocks.append(stage_blocks)
            if stage < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim *= 2
                height, width = height // 2, width // 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._spatial = (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        merge_idx = 0
        for stage_idx, stage_blocks in enumerate(self.blocks):
            for block in stage_blocks:
                x = block(x, height, width)
            if stage_idx < len(self.merges):
                x, height, width = self.merges[merge_idx](x, height, width)
                merge_idx += 1
        x = self.norm(x).mean(dim=1)
        return self.head(x)
