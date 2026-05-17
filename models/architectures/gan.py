"""GAN architectures sized for 28x28 MNIST."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel

MNIST_IMAGE_SIZE = 28


def _ensure_image_4d(x: torch.Tensor, image_size: int = MNIST_IMAGE_SIZE) -> torch.Tensor:
    """Convert batch to ``(N, 1, H, W)``."""
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x.view(x.size(0), 1, image_size, image_size)


def _build_mnist_generator(latent_dim: int, generator_channels: int, out_channels: int = 1) -> nn.Sequential:
    """Transposed-conv generator targeting 28x28 output."""
    gc = generator_channels
    return nn.Sequential(
        nn.ConvTranspose2d(latent_dim, gc * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(gc * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(gc * 8, gc * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(gc * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(gc * 4, gc * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(gc * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(gc * 2, out_channels, 4, 2, 1, bias=False),
        nn.Tanh(),
    )


def _build_mnist_discriminator(
    in_channels: int,
    discriminator_channels: int,
    use_sigmoid: bool = True,
) -> nn.Sequential:
    """Conv discriminator for 28x28 inputs (3x3 kernels, no spatial underflow)."""
    dc = discriminator_channels
    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, dc, 3, 1, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(dc, dc * 2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(dc * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(dc * 2, dc * 4, 3, 2, 1, bias=False),
        nn.BatchNorm2d(dc * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(dc * 4, dc * 8, 3, 2, 1, bias=False),
        nn.BatchNorm2d(dc * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(dc * 8, 1, 3, 1, 0, bias=False),
        nn.AdaptiveAvgPool2d(1),
    ]
    if use_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class GANBase(BaseModel):
    """Base class for GAN models."""

    def __init__(self, num_classes: int = 10, latent_dim: int = 100, image_size: int = MNIST_IMAGE_SIZE, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

    def _resize_generated(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-2:] != (self.image_size, self.image_size):
            img = F.interpolate(
                img,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return img

    def _discriminator_output(self, out: torch.Tensor) -> torch.Tensor:
        """Return per-sample discriminator scores as ``(N, 1)``."""
        return out.view(out.size(0), -1)


class VanillaGAN(GANBase):
    """Vanilla GAN implementation (fully connected)."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 100,
        generator_hidden: int = 256,
        discriminator_hidden: int = 256,
        image_size: int = MNIST_IMAGE_SIZE,
    ):
        super().__init__(num_classes=num_classes, latent_dim=latent_dim, image_size=image_size)
        pixels = image_size * image_size

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, generator_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(generator_hidden, generator_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(generator_hidden * 2, pixels),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pixels, discriminator_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden * 2, discriminator_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_image_4d(x, self.image_size)
        return self.discriminator(x.view(x.size(0), -1))

    def generate(self, z: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        if z is None:
            z = torch.randn(batch_size, self.latent_dim)
        out = self.generator(z)
        return out.view(-1, 1, self.image_size, self.image_size)


class DCGAN(GANBase):
    """Deep Convolutional GAN for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 100,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        image_size: int = MNIST_IMAGE_SIZE,
    ):
        super().__init__(num_classes=num_classes, latent_dim=latent_dim, image_size=image_size)
        self.generator = _build_mnist_generator(latent_dim, generator_channels)
        self.discriminator = _build_mnist_discriminator(1, discriminator_channels, use_sigmoid=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_image_4d(x, self.image_size)
        return self._discriminator_output(self.discriminator(x))

    def generate(self, z: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1)
        return self._resize_generated(self.generator(z))


class WGAN(GANBase):
    """Wasserstein GAN for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 100,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        image_size: int = MNIST_IMAGE_SIZE,
    ):
        super().__init__(num_classes=num_classes, latent_dim=latent_dim, image_size=image_size)
        self.generator = _build_mnist_generator(latent_dim, generator_channels)
        self.discriminator = _build_mnist_discriminator(1, discriminator_channels, use_sigmoid=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_image_4d(x, self.image_size)
        return self._discriminator_output(self.discriminator(x))

    def generate(self, z: Optional[torch.Tensor] = None, batch_size: int = 1) -> torch.Tensor:
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1)
        return self._resize_generated(self.generator(z))


class CGAN(GANBase):
    """Conditional GAN for MNIST."""

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 100,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        image_size: int = MNIST_IMAGE_SIZE,
    ):
        super().__init__(num_classes=num_classes, latent_dim=latent_dim, image_size=image_size)
        self.generator = _build_mnist_generator(latent_dim + num_classes, generator_channels)
        self.discriminator = _build_mnist_discriminator(
            1 + num_classes, discriminator_channels, use_sigmoid=True
        )

    def _embed_labels(self, labels: torch.Tensor, spatial: tuple[int, int]) -> torch.Tensor:
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
        one_hot = one_hot.view(labels.size(0), self.num_classes, 1, 1)
        return one_hot.expand(-1, -1, spatial[0], spatial[1])

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = _ensure_image_4d(x, self.image_size)
        if labels is not None:
            x = torch.cat([x, self._embed_labels(labels, x.shape[-2:])], dim=1)
        return self._discriminator_output(self.discriminator(x))

    def generate(
        self,
        z: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=labels.device if labels is not None else None)
        if labels is not None:
            one_hot = self._embed_labels(labels, (1, 1))
            z = torch.cat([z, one_hot], dim=1)
        return self._resize_generated(self.generator(z))
