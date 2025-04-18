import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class WGANtrainer:
    def __init__(self, model, device, learning_rate=0.00005, n_critic=5, clip_value=0.01):
        self.model = model
        self.device = device
        self.n_critic = n_critic  # Number of D updates per G update
        self.clip_value = clip_value
        
        # Separate optimizers for generator and discriminator
        self.g_optimizer = optim.RMSprop(model.generator.parameters(), lr=learning_rate)
        self.d_optimizer = optim.RMSprop(model.discriminator.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for real_images, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Train Discriminator (Critic)
            for _ in range(self.n_critic):
                self.d_optimizer.zero_grad()
                
                # Real images
                d_real = self.model.discriminator(real_images)
                d_real_loss = -torch.mean(d_real)  # WGAN loss for real images
                
                # Fake images
                z = torch.randn(batch_size, self.model.latent_dim, 1, 1).to(self.device)
                fake_images = self.model.generator(z)
                d_fake = self.model.discriminator(fake_images.detach())
                d_fake_loss = torch.mean(d_fake)  # WGAN loss for fake images
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Clip weights of discriminator
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)
                
                running_d_loss += d_loss.item()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            fake_images = self.model.generator(z)
            g_output = self.model.discriminator(fake_images)
            g_loss = -torch.mean(g_output)  # WGAN loss for generator
            
            g_loss.backward()
            self.g_optimizer.step()
            
            running_g_loss += g_loss.item()
            
            pbar.set_postfix({
                'g_loss': running_g_loss/len(pbar),
                'd_loss': running_d_loss/(len(pbar) * self.n_critic)
            })
            
        return running_g_loss/len(train_loader), running_d_loss/(len(train_loader) * self.n_critic)
    
    def save_checkpoint(self, path, epoch, g_loss, d_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
        }, path) 