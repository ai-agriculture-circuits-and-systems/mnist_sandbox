import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.training_factory import get_optimizer


def _is_fc_gan(model: nn.Module) -> bool:
    """True for fully-connected VanillaGAN (Linear generator)."""
    generator = getattr(model, "generator", None)
    if generator is None:
        return False
    if isinstance(generator, nn.Sequential) and len(generator) > 0:
        return isinstance(generator[0], nn.Linear)
    return isinstance(generator, nn.Linear)


class GANTrainer:
    def __init__(self, model, device, learning_rate=0.001, optimizer_name="adam"):
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.fc_gan = _is_fc_gan(model)

        self.g_optimizer = get_optimizer(
            optimizer_name, model.generator.parameters(), learning_rate
        )
        self.d_optimizer = get_optimizer(
            optimizer_name, model.discriminator.parameters(), learning_rate
        )

    def _flatten_real(self, real_images: torch.Tensor, batch_size: int) -> torch.Tensor:
        if real_images.dim() == 4:
            return real_images.view(batch_size, -1)
        return real_images

    def train_epoch(self, train_loader):
        self.model.train()
        running_g_loss = 0.0
        running_d_loss = 0.0

        pbar = tqdm(train_loader, desc='Training')
        for real_images, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            self.d_optimizer.zero_grad()

            if self.fc_gan:
                real_flat = self._flatten_real(real_images, batch_size)
                d_real = self.model.discriminator(real_flat)
                d_real = d_real.view(batch_size, 1)

                z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
                fake_flat = self.model.generator(z)
                d_fake = self.model.discriminator(fake_flat.detach())
                d_fake = d_fake.view(batch_size, 1)
            else:
                if len(real_images.shape) == 2:
                    real_images = real_images.view(batch_size, 1, 28, 28)
                d_real = self.model.discriminator(real_images)
                d_real = d_real.view(batch_size, 1)

                z = torch.randn(batch_size, self.model.latent_dim, 1, 1, device=self.device)
                fake_images = self.model.generator(z)
                d_fake = self.model.discriminator(fake_images.detach())
                d_fake = d_fake.view(batch_size, 1)

            d_loss = self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)
            d_loss.backward()
            self.d_optimizer.step()

            self.g_optimizer.zero_grad()

            if self.fc_gan:
                z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
                fake_flat = self.model.generator(z)
                g_output = self.model.discriminator(fake_flat).view(batch_size, 1)
            else:
                fake_images = self.model.generator(z)
                g_output = self.model.discriminator(fake_images).view(batch_size, 1)

            g_loss = self.criterion(g_output, real_labels)
            g_loss.backward()
            self.g_optimizer.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()

            pbar.set_postfix({
                'g_loss': running_g_loss / len(pbar),
                'd_loss': running_d_loss / len(pbar),
            })

        return running_g_loss / len(train_loader), running_d_loss / len(train_loader)

    def save_checkpoint(self, path, epoch, g_loss, d_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
        }, path)
