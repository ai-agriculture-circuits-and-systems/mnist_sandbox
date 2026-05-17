"""ViM-style vision backbone using bidirectional state-space inspired conv blocks."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class SSMLikeBlock(nn.Module):
    """Gated depthwise conv block approximating long-range mixing."""

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dw_fwd = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.dw_bwd = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        fwd = self.dw_fwd(x)
        bwd = torch.flip(self.dw_bwd(torch.flip(x, dims=[3])), dims=[3])
        mixed = fwd + bwd
        x = mixed * self.gate(x)
        return residual + self.proj(x)


class VimTiny(BaseModel):
    """ViM-tiny style model for 224x224 input."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 224,
        embed_dim: int = 128,
        depths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        depths = depths or [2, 4, 6, 2]
        layers: List[nn.Module] = [
            nn.Conv2d(1, embed_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        ]
        dim = embed_dim
        for stage, depth in enumerate(depths):
            if stage > 0:
                layers.extend(
                    [
                        nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False),
                        nn.BatchNorm2d(dim * 2),
                        nn.SiLU(inplace=True),
                    ]
                )
                dim *= 2
            for _ in range(depth):
                layers.append(SSMLikeBlock(dim))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
