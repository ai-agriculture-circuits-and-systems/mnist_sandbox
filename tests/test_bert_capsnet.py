"""Tests for BERT and CapsNet input-size handling."""

from __future__ import annotations

import torch

from models.architectures.bert import BERTMNIST
from models.architectures.capsnet import CapsNet, primary_spatial_size


def test_primary_spatial_size_mnist() -> None:
    """Two 9x9 valid convs on 28x28 yield 12x12 maps."""
    assert primary_spatial_size(28) == 12


def test_primary_spatial_size_224() -> None:
    """Two 9x9 valid convs on 224x224 yield 208x208 maps."""
    assert primary_spatial_size(224) == 208


def test_bert_pos_embedding_matches_max_seq_length() -> None:
    """Positional embedding length follows max_seq_length (e.g. 224x224 -> 50176)."""
    model = BERTMNIST(
        num_classes=6,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_seq_length=50176,
    )
    assert model.pos_embedding.shape[1] == 50176


def test_bert_forward_28() -> None:
    """BERT forward pass on MNIST-sized input."""
    model = BERTMNIST(
        num_classes=10,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_seq_length=784,
    )
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_capsnet_route_weights_fixed_grid() -> None:
    """Routing params use a fixed route_spatial grid, independent of input_size."""
    small = CapsNet(num_classes=6, input_size=28, route_spatial=12)
    large = CapsNet(num_classes=6, input_size=224, route_spatial=12)
    assert small.route_weights.shape[1] == 32 * 12 * 12
    assert large.route_weights.shape[1] == 32 * 12 * 12


def test_capsnet_forward_224() -> None:
    """CapsNet forward pass on PlantVillage-sized input."""
    model = CapsNet(num_classes=6, input_size=224, primary_caps=16, route_spatial=12)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)


def test_capsnet_forward_28() -> None:
    """CapsNet forward pass on MNIST-sized input."""
    model = CapsNet(num_classes=10, input_size=28)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)
