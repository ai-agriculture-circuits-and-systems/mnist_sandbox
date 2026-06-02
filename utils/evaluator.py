import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def macro_f1_score(targets, preds) -> float:
    """Compute macro-averaged F1 (%) over classes present in ``targets``."""
    if len(targets) == 0:
        return 0.0
    return float(
        f1_score(targets, preds, average="macro", zero_division=0) * 100.0
    )


class Evaluator:
    def __init__(self, model, device, loss_name="cross_entropy"):
        self.model = model
        self.device = device
        from utils.training_factory import get_classification_loss

        self.criterion = get_classification_loss(loss_name)
        
    def evaluate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

        macro_f1 = macro_f1_score(all_targets, all_preds)
        return running_loss / len(test_loader), 100.0 * correct / total, macro_f1, all_preds, all_targets
    
    def plot_confusion_matrix(self, all_preds, all_targets, save_path=None):
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
        plt.close() 