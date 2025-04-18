import torch
import torch.nn as nn
from ..base_model import BaseModel

class AutoencoderBase(BaseModel):
    """Base class for autoencoder models"""
    def __init__(self, num_classes=10, latent_dim=32, **kwargs):
        super(AutoencoderBase, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
    def forward(self, x):
        # This is a base class, not meant to be used directly
        raise NotImplementedError("Subclasses must implement forward method")
    
    def encode(self, x):
        # This is a base class, not meant to be used directly
        raise NotImplementedError("Subclasses must implement encode method")
    
    def decode(self, z):
        # This is a base class, not meant to be used directly
        raise NotImplementedError("Subclasses must implement decode method")

class SimpleAutoencoder(AutoencoderBase):
    """Simple fully connected autoencoder"""
    def __init__(self, num_classes=10, latent_dim=32, hidden_dims=[128, 64]):
        super(SimpleAutoencoder, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Encoder
        encoder_layers = []
        input_dim = 784  # 28x28 = 784 pixels
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, 784))
        decoder_layers.append(nn.Sigmoid())  # Output between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder(z)
        # Reshape to original dimensions
        return x.view(x.size(0), 1, 28, 28)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class ConvolutionalAutoencoder(AutoencoderBase):
    """Convolutional autoencoder"""
    def __init__(self, num_classes=10, latent_dim=32, channels=[32, 64, 128]):
        super(ConvolutionalAutoencoder, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Encoder
        encoder_layers = []
        in_channels = 1  # MNIST is grayscale
        
        # Calculate feature map sizes based on input size
        self.feature_sizes = []
        current_size = 224  # Handle any input size
        
        for out_channels in channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ])
            current_size = current_size // 2
            self.feature_sizes.append(current_size)
            in_channels = out_channels
        
        # Calculate the size of the flattened feature map
        self.flatten_size = channels[-1] * self.feature_sizes[-1] * self.feature_sizes[-1]
        
        # Add flatten and linear layer for latent space
        encoder_layers.extend([
            nn.Flatten(),
            nn.Linear(self.flatten_size, latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = [
            nn.Linear(latent_dim, self.flatten_size),
            nn.ReLU()
        ]
        
        # Unflatten to match the last encoder feature map
        decoder_layers.append(nn.Unflatten(1, (channels[-1], self.feature_sizes[-1], self.feature_sizes[-1])))
        
        # Reverse the channels for the decoder
        for i in range(len(channels) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ])
        
        # Final layer to get back to original dimensions
        decoder_layers.extend([
            nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class VariationalAutoencoder(AutoencoderBase):
    """Variational Autoencoder (VAE)"""
    def __init__(self, num_classes=10, latent_dim=32, hidden_dims=[128, 64]):
        super(VariationalAutoencoder, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        
        # Calculate input dimension based on image size
        self.input_dim = 1  # Start with number of channels
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        # Use convolutional layers instead of linear
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ])
            in_channels = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate size of flattened features
        self.flatten_size = hidden_dims[-1] * (224 // (2 ** len(hidden_dims))) ** 2
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        decoder_layers = []
        
        # Initial linear layer to get back to convolutional shape
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # Use transposed convolutions for upsampling
        in_channels = hidden_dims[-1]
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i-1]),
                nn.ReLU()
            ])
        
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_dims[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[-1], self.feature_size, self.feature_size)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DenoisingAutoencoder(AutoencoderBase):
    """Denoising Autoencoder"""
    def __init__(self, num_classes=10, latent_dim=32, hidden_dims=[128, 64], noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__(num_classes=num_classes, latent_dim=latent_dim)
        self.noise_factor = noise_factor
        
        # Use convolutional layers instead of linear
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ])
            in_channels = hidden_dim
        
        # Calculate feature map size
        self.feature_size = 224 // (2 ** len(hidden_dims))
        self.flatten_size = hidden_dims[-1] * self.feature_size * self.feature_size
        
        encoder_layers.extend([
            nn.Flatten(),
            nn.Linear(self.flatten_size, latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.ReLU()
        )
        
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i-1]),
                nn.ReLU()
            ])
        
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_dims[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[-1], self.feature_size, self.feature_size)
        return self.decoder(x)
    
    def forward(self, x):
        noisy_x = self.add_noise(x)
        z = self.encode(noisy_x)
        return self.decode(z) 