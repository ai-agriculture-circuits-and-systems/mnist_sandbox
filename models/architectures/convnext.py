"""ConvNeXt-style modern CNN."""

from typing import List, Optional

import torch
import torch.nn as nn

from ..base_model import BaseModel


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div(keep)
        return x * mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7 + LayerNorm + pointwise MLP."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXt(BaseModel):
    """Tiny ConvNeXt for grayscale MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        depths = depths or [2, 2, 4, 2]
        dims = dims or [48, 96, 192, 384]

        total_blocks = sum(depths)
        dpr = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

        self.stem = nn.Conv2d(1, dims[0], kernel_size=4, stride=4)

        stages: List[nn.Module] = []
        dp_idx = 0
        for i in range(4):
            if i > 0:
                stages.append(nn.Sequential(
                    nn.BatchNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                ))
            blocks = []
            for _ in range(depths[i]):
                blocks.append(ConvNeXtBlock(dims[i], dpr[dp_idx]))
                dp_idx += 1
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        return self.head(x)
