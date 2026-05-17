"""Simplified GoogLeNet (Inception) for MNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class InceptionModule(nn.Module):
    """Inception module with parallel conv branches."""

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            F.relu(self.branch1(x), inplace=True),
            F.relu(self.branch2(x), inplace=True),
            F.relu(self.branch3(x), inplace=True),
            F.relu(self.branch4(x), inplace=True),
        ]
        return torch.cat(branches, dim=1)


class GoogLeNet(BaseModel):
    """Lightweight GoogLeNet-style classifier for grayscale images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3a = InceptionModule(32, 16, 16, 32, 4, 8, 8)
        self.inception3b = InceptionModule(64, 32, 32, 64, 8, 16, 16)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(128, 32, 48, 96, 8, 16, 16)
        self.inception4b = InceptionModule(160, 48, 64, 128, 12, 24, 24)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(224, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
