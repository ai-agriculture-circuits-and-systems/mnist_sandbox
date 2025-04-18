import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, num_classes=10, hidden_sizes=[512, 256, 128]):
        super(MLP, self).__init__()
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Calculate input size (28x28 = 784 for MNIST)
        input_size = 28 * 28
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x 