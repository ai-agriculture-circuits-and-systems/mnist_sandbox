import torch
import torch.nn as nn
from ..base_model import BaseModel

class GANBase(BaseModel):
    """Base class for GAN models"""
    def __init__(self, num_classes=10, latent_dim=100, **kwargs):
        super(GANBase, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
    def forward(self, x):
        # This is a base class, not meant to be used directly
        raise NotImplementedError("Subclasses must implement forward method")

class VanillaGAN(GANBase):
    """Vanilla GAN implementation"""
    def __init__(self, num_classes=10, latent_dim=100, generator_hidden=256, discriminator_hidden=256):
        super(VanillaGAN, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, generator_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(generator_hidden, generator_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(generator_hidden * 2, 784),  # 28x28 = 784 pixels
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(784, discriminator_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden * 2, discriminator_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, 784)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 784)
        return self.discriminator(x)
    
    def generate(self, z=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
        return self.generator(z)

class DCGAN(GANBase):
    """Deep Convolutional GAN implementation"""
    def __init__(self, num_classes=10, latent_dim=100, generator_channels=64, discriminator_channels=64):
        super(DCGAN, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Generator
        self.generator = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, generator_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_channels * 8),
            nn.ReLU(True),
            # State size: (generator_channels * 8) x 4 x 4
            nn.ConvTranspose2d(generator_channels * 8, generator_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 4),
            nn.ReLU(True),
            # State size: (generator_channels * 4) x 8 x 8
            nn.ConvTranspose2d(generator_channels * 4, generator_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 2),
            nn.ReLU(True),
            # State size: (generator_channels * 2) x 16 x 16
            nn.ConvTranspose2d(generator_channels * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: 1 x 28 x 28
        )
        
        # Discriminator - Completely redesigned for MNIST
        self.discriminator = nn.Sequential(
            # Input is 1 x 28 x 28
            nn.Conv2d(1, discriminator_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: discriminator_channels x 28 x 28
            
            nn.Conv2d(discriminator_channels, discriminator_channels * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 2) x 14 x 14
            
            nn.Conv2d(discriminator_channels * 2, discriminator_channels * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 4) x 7 x 7
            
            nn.Conv2d(discriminator_channels * 4, discriminator_channels * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 8) x 4 x 4
            
            nn.Conv2d(discriminator_channels * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape input from (batch_size, 784) to (batch_size, 1, 28, 28)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        return self.discriminator(x)
    
    def generate(self, z=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        return self.generator(z)

class WGAN(GANBase):
    """Wasserstein GAN implementation"""
    def __init__(self, num_classes=10, latent_dim=100, generator_channels=64, discriminator_channels=64):
        super(WGAN, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Generator (similar to DCGAN)
        self.generator = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, generator_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_channels * 8),
            nn.ReLU(True),
            # State size: (generator_channels * 8) x 4 x 4
            nn.ConvTranspose2d(generator_channels * 8, generator_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 4),
            nn.ReLU(True),
            # State size: (generator_channels * 4) x 8 x 8
            nn.ConvTranspose2d(generator_channels * 4, generator_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 2),
            nn.ReLU(True),
            # State size: (generator_channels * 2) x 16 x 16
            nn.ConvTranspose2d(generator_channels * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: 1 x 28 x 28
        )
        
        # Discriminator (Critic in WGAN)
        self.discriminator = nn.Sequential(
            # Input is 1 x 28 x 28
            nn.Conv2d(1, discriminator_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: discriminator_channels x 14 x 14
            nn.Conv2d(discriminator_channels, discriminator_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 2) x 7 x 7
            nn.Conv2d(discriminator_channels * 2, discriminator_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 4) x 4 x 4
            nn.Conv2d(discriminator_channels * 4, 1, 4, 1, 0, bias=False)
            # No sigmoid for WGAN
        )
    
    def forward(self, x):
        # Reshape input from (batch_size, 784) to (batch_size, 1, 28, 28)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        return self.discriminator(x)
    
    def generate(self, z=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        return self.generator(z)

class CGAN(GANBase):
    """Conditional GAN implementation"""
    def __init__(self, num_classes=10, latent_dim=100, generator_channels=64, discriminator_channels=64):
        super(CGAN, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Generator
        self.generator = nn.Sequential(
            # Input is latent_dim + num_classes x 1 x 1
            nn.ConvTranspose2d(latent_dim + num_classes, generator_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_channels * 8),
            nn.ReLU(True),
            # State size: (generator_channels * 8) x 4 x 4
            nn.ConvTranspose2d(generator_channels * 8, generator_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 4),
            nn.ReLU(True),
            # State size: (generator_channels * 4) x 8 x 8
            nn.ConvTranspose2d(generator_channels * 4, generator_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_channels * 2),
            nn.ReLU(True),
            # State size: (generator_channels * 2) x 16 x 16
            nn.ConvTranspose2d(generator_channels * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: 1 x 28 x 28
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # Input is 1 + num_classes x 28 x 28
            nn.Conv2d(1 + num_classes, discriminator_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: discriminator_channels x 14 x 14
            nn.Conv2d(discriminator_channels, discriminator_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 2) x 7 x 7
            nn.Conv2d(discriminator_channels * 2, discriminator_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (discriminator_channels * 4) x 4 x 4
            nn.Conv2d(discriminator_channels * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels=None):
        # Reshape input from (batch_size, 784) to (batch_size, 1, 28, 28)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        
        if labels is not None:
            # One-hot encode labels
            one_hot = torch.zeros(labels.size(0), self.num_classes, device=x.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            # Reshape one_hot to match image dimensions
            one_hot = one_hot.view(one_hot.size(0), one_hot.size(1), 1, 1)
            one_hot = one_hot.expand(-1, -1, x.size(2), x.size(3))
            # Concatenate with input
            x = torch.cat([x, one_hot], 1)
        return self.discriminator(x)
    
    def generate(self, z=None, labels=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        
        if labels is not None:
            # One-hot encode labels
            one_hot = torch.zeros(labels.size(0), self.num_classes, device=z.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            # Reshape one_hot to match z dimensions
            one_hot = one_hot.view(one_hot.size(0), one_hot.size(1), 1, 1)
            # Concatenate with z
            z = torch.cat([z, one_hot], 1)
        
        return self.generator(z) 