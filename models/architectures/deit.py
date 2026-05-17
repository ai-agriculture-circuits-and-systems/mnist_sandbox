"""DeiT: Data-efficient Image Transformer with distillation token."""

import torch
import torch.nn as nn

from ..base_model import BaseModel
from .vit import PatchEmbed, TransformerBlock


class DeiT(BaseModel):
    """DeiT with class and distillation tokens (224x224 default)."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        dist_tokens = self.dist_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])
        return (cls_out + dist_out) / 2
