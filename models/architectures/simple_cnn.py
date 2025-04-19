import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
import torch

class SimpleCNN(BaseModel):
    def __init__(self, num_classes=10, channels=[32, 64, 64], input_size=224):
        super(SimpleCNN, self).__init__()
        
        # Ensure we have at least 3 channels
        if len(channels) < 3:
            channels = channels + [channels[-1]] * (3 - len(channels))
        
        # Convolutional layers with batch normalization
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels)):
            in_channels = 1 if i == 0 else channels[i-1]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(inplace=True)
            ))
        
        # Pooling layers
        self.pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(channels))
        ])
        
        # Calculate size of flattened features dynamically
        with torch.no_grad():
            # Create a dummy input to calculate the output size
            x = torch.zeros(1, 1, input_size, input_size)
            for conv, pool in zip(self.conv_layers, self.pool_layers):
                x = pool(conv(x))
            self.flat_features = x.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with pooling
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = pool(conv(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x 