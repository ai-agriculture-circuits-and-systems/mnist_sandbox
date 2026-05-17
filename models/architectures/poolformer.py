"""PoolFormer (MetaFormer) using average pooling as token mixer."""

from typing import List, Optional

import torch
import torch.nn as nn

from ..base_model import BaseModel


class PoolFormerBlock(nn.Module):
    """Transformer-style block with pooling instead of self-attention."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PoolFormer(BaseModel):
    """PoolFormer-S12 style hierarchy adapted for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        dims = dims or [32, 64, 128, 256]
        depths = depths or [2, 2, 4, 2]

        layers: List[nn.Module] = [
            nn.Conv2d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, dims[0]),
        ]
        in_ch = dims[0]
        for stage, (dim, depth) in enumerate(zip(dims, depths)):
            if stage > 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, dim, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.GroupNorm(1, dim),
                    )
                )
                in_ch = dim
            for _ in range(depth):
                layers.append(PoolFormerBlock(in_ch, drop=drop))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
