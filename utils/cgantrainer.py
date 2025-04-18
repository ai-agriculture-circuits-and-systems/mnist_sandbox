import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class CGANTrainer:
    def __init__(self, model, device, learning_rate=0.0002, beta1=0.5):
        self.model = model
        self.device = device
        
        # Separate optimizers for generator and discriminator
        self.g_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for real_images, labels in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            labels = labels.to(self.device)
            
            # Create labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real images
            d_real = self.model(real_images, labels)
            d_real_loss = self.criterion(d_real, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, self.model.latent_dim, 1, 1).to(self.device)
            fake_images = self.model.generate(z, labels)
            d_fake = self.model(fake_images.detach(), labels)
            d_fake_loss = self.criterion(d_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            # Generate fake images again since we need to update generator
            fake_images = self.model.generate(z, labels)
            g_output = self.model(fake_images, labels)
            g_loss = self.criterion(g_output, real_labels)  # We want generator to fool discriminator
            
            g_loss.backward()
            self.g_optimizer.step()
            
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'g_loss': running_g_loss/len(pbar),
                'd_loss': running_d_loss/len(pbar)
            })
            
        return running_g_loss/len(train_loader), running_d_loss/len(train_loader)
    
    def save_checkpoint(self, path, epoch, g_loss, d_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
        }, path) 