import torch
from tqdm import tqdm
from models.architectures.gan import CGAN
from utils.training_factory import get_classification_loss, get_optimizer

class Trainer:
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        loss_name="cross_entropy",
        optimizer_name="adam",
        weight_decay=0.0,
    ):
        self.model = model
        self.device = device
        self.criterion = get_classification_loss(loss_name)
        self.optimizer = get_optimizer(
            optimizer_name, model.parameters(), learning_rate, weight_decay
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, CGAN):
                outputs = self.model(inputs, targets)
            else:
                outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            running_acc += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/len(pbar),
                'acc': 100.*running_acc/total
            })
        
        return running_loss/len(train_loader), 100.*running_acc/total
    
    def save_checkpoint(self, path, epoch, loss, acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
        }, path)