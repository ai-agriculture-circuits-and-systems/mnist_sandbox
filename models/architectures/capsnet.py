"""Simplified Capsule Network for MNIST digit classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Squash activation along capsule dimension."""
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)


def primary_spatial_size(input_size: int, kernel_size: int = 9, num_convs: int = 2) -> int:
    """Return H=W feature map size after stacked valid conv layers."""
    size = input_size
    for _ in range(num_convs):
        size = size - kernel_size + 1
    if size < 1:
        raise ValueError(
            f"input_size={input_size} is too small for {num_convs} "
            f"conv layers with kernel_size={kernel_size}"
        )
    return size


class CapsNet(BaseModel):
    """CapsNet with dynamic routing between primary and digit capsules."""

    def __init__(
        self,
        num_classes: int = 10,
        primary_caps: int = 32,
        primary_dim: int = 8,
        digit_dim: int = 16,
        routing_iters: int = 3,
        input_size: int = 28,
        route_spatial: int = 12,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.primary_caps = primary_caps
        self.primary_dim = primary_dim
        self.digit_dim = digit_dim
        self.routing_iters = routing_iters
        self.input_size = input_size
        self.route_spatial = route_spatial

        # Pool primary maps to a fixed grid so routing params do not grow with input_size^2.
        num_routes = primary_caps * route_spatial * route_spatial

        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9),
            nn.ReLU(inplace=True),
        )
        self.primary = nn.Conv2d(256, primary_caps * primary_dim, kernel_size=9)
        self.route_weights = nn.Parameter(
            torch.randn(1, num_routes, num_classes, digit_dim, primary_dim) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.primary(x)
        x = F.adaptive_avg_pool2d(x, (self.route_spatial, self.route_spatial))
        batch = x.size(0)
        u = x.view(batch, self.primary_caps, self.primary_dim, self.route_spatial, self.route_spatial)
        u = u.permute(0, 1, 3, 4, 2).reshape(
            batch, self.primary_caps * self.route_spatial * self.route_spatial, self.primary_dim
        )
        u = squash(u, dim=-1)

        weights = self.route_weights[0, : u.size(1)]
        u_hat = torch.einsum("bni,ncji->bncj", u, weights)

        b = torch.zeros(batch, u.size(1), self.num_classes, device=x.device)
        for iteration in range(self.routing_iters):
            c = F.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)
            if iteration < self.routing_iters - 1:
                b = b + torch.sum(u_hat * v.unsqueeze(1), dim=-1)

        return torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
