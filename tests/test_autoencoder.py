"""Tests for autoencoder input-size handling."""

from __future__ import annotations

import torch

from models.architectures.autoencoder import SimpleAutoencoder, VariationalAutoencoder
from utils.autoencoder_trainer import AutoencoderTrainer, _prepare_bce_reconstruction_pair


def test_simple_autoencoder_accepts_configured_input_size() -> None:
    """SimpleAutoencoder must match flattened input dim to image size."""
    model = SimpleAutoencoder(input_size=224, channels=1, hidden_dims=[64, 32], latent_dim=16)
    batch = torch.rand(2, 1, 224, 224)
    recon = model(batch)
    assert recon.shape == batch.shape


def test_simple_autoencoder_default_mnist_size() -> None:
    """Default 28x28 configuration remains backward compatible."""
    model = SimpleAutoencoder()
    batch = torch.rand(4, 1, 28, 28)
    recon = model(batch)
    assert recon.shape == batch.shape


def test_prepare_bce_clamps_out_of_range_values() -> None:
    """BCE prep must keep tensors inside (0, 1) even with bad inputs."""
    recon = torch.tensor([[-1.0, 2.0], [float("nan"), 0.5]])
    targets = torch.tensor([[0.0, 1.0], [0.5, 1.5]])
    recon_out, targets_out = _prepare_bce_reconstruction_pair(recon, targets)
    assert torch.isfinite(recon_out).all()
    assert torch.isfinite(targets_out).all()
    assert recon_out.min() > 0 and recon_out.max() < 1
    assert targets_out.min() > 0 and targets_out.max() < 1


def test_variational_autoencoder_bce_forward() -> None:
    """VariationalAutoencoder with BCE must not use the wrong VAE class name check."""
    model = VariationalAutoencoder(input_size=28, latent_dim=8, hidden_dims=[32, 16])
    trainer = AutoencoderTrainer(
        model,
        device=torch.device("cpu"),
        loss_name="bce",
        optimizer_name="adam",
    )
    batch = torch.rand(2, 1, 28, 28)
    outputs = model(batch)
    loss = trainer._compute_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
