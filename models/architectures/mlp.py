import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, num_classes=10, hidden_sizes=[2048, 1024, 512, 256], input_size=224):
        super(MLP, self).__init__()
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Calculate input size
        self.input_size = input_size * input_size
        
        # Create layers
        layers = []
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))
            # Dropout
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input
        x = self.flatten(x)
        
        # Apply layers
        x = self.layers(x)
        
        return x 