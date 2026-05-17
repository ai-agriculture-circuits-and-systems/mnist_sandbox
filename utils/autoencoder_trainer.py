"""Training and evaluation for autoencoder models."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.training_factory import get_autoencoder_loss, get_optimizer


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

    def _compute_loss(self, outputs: torch.Tensor | tuple, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            recon, mu, logvar = outputs
            recon_loss = self.criterion(recon, inputs)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + 0.001 * kl
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
        self.criterion = get_autoencoder_loss(loss_name)

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
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / len(pbar)})

        avg_loss = running_loss / len(test_loader)
        return avg_loss, 0.0, [], []
