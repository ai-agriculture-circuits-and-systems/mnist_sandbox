"""Training and evaluation for autoencoder models."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.training_factory import get_autoencoder_loss, get_optimizer


def _uses_bce_loss(loss_name: str) -> bool:
    """Return True when the criterion is binary cross-entropy."""
    return loss_name.lower() in {"bce", "binary_cross_entropy"}


def _clamp_bce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sanitize values for BCE: finite and strictly inside [0, 1]."""
    tensor = torch.nan_to_num(tensor, nan=0.5, posinf=1.0, neginf=0.0)
    return torch.clamp(tensor, 1e-6, 1.0 - 1e-6)


def _prepare_bce_reconstruction_pair(
    recon: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map reconstructions and targets into the valid range for BCELoss."""
    if recon.min() < 0 or recon.max() > 1:
        recon = torch.sigmoid(recon)
    return _clamp_bce_tensor(recon), _clamp_bce_tensor(targets)


class AutoencoderTrainer:
    """Trainer for reconstruction-based autoencoder models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        loss_name: str = "mse",
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.device = device
        self.criterion = get_autoencoder_loss(loss_name)
        self.optimizer = get_optimizer(
            optimizer_name, model.parameters(), learning_rate, weight_decay
        )
        self.loss_name = loss_name

    def _prepare_reconstruction_targets(
        self, recon: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare reconstruction and targets for the active loss."""
        if _uses_bce_loss(self.loss_name):
            return _prepare_bce_reconstruction_pair(recon, targets)
        return recon, targets

    def _compute_loss(self, outputs: torch.Tensor | tuple, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            recon, mu, logvar = outputs
            recon, inputs = self._prepare_reconstruction_targets(recon, inputs)
            recon_loss = self.criterion(recon, inputs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + 0.001 * kl
        outputs, inputs = self._prepare_reconstruction_targets(outputs, inputs)
        return self.criterion(outputs, inputs)

    def train_epoch(self, train_loader) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc="Training AE")
        for inputs, _ in pbar:
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / len(pbar)})

        return running_loss / len(train_loader), 0.0


class AutoencoderEvaluator:
    """Evaluator for autoencoder reconstruction loss."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_name: str = "mse",
    ) -> None:
        self.model = model
        self.device = device
        self.loss_name = loss_name
        self.criterion = get_autoencoder_loss(loss_name)

    def _clamp_bce_pair(
        self, outputs: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _uses_bce_loss(self.loss_name):
            return outputs, inputs
        return _prepare_bce_reconstruction_pair(outputs, inputs)

    def evaluate(self, test_loader) -> tuple[float, float, list, list]:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating AE")
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs, inputs = self._clamp_bce_pair(outputs, inputs)
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / len(pbar)})

        avg_loss = running_loss / len(test_loader)
        return avg_loss, 0.0, [], []
