"""Factories for loss functions and optimizers used in training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


CLASSIFICATION_LOSSES = ("cross_entropy", "label_smoothing", "focal_loss")
AUTOENCODER_LOSSES = ("mse", "l1", "bce")
GAN_LOSSES = ("bce", "wasserstein")
OPTIMIZERS = ("adam", "sgd", "adamw", "rmsprop")


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def get_classification_loss(loss_name: str) -> nn.Module:
    """Return a classification loss module."""
    losses = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        "focal_loss": FocalLoss(),
    }
    if loss_name not in losses:
        raise ValueError(
            f"Unknown classification loss '{loss_name}'. "
            f"Choose from: {list(losses.keys())}"
        )
    return losses[loss_name]


def get_autoencoder_loss(loss_name: str) -> nn.Module:
    """Return a reconstruction loss module."""
    losses = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "bce": nn.BCELoss(),
    }
    if loss_name not in losses:
        raise ValueError(
            f"Unknown autoencoder loss '{loss_name}'. Choose from: {list(losses.keys())}"
        )
    return losses[loss_name]


def get_optimizer(
    optimizer_name: str,
    parameters: Any,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    """Return an optimizer for the given parameters."""
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    if name == "adamw":
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose from: {OPTIMIZERS}")
