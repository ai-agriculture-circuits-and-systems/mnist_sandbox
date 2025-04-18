import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
import torch

class SimpleCNN(BaseModel):
    def __init__(self, num_classes=10, channels=[32, 64, 64], input_size=28):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size of flattened features dynamically
        with torch.no_grad():
            # Create a dummy input to calculate the output size
            x = torch.zeros(1, 1, input_size, input_size)
            x = self.pool(F.relu(self.conv1(x)))  # After conv1 + pool
            x = self.pool(F.relu(self.conv2(x)))  # After conv2 + pool
            x = self.pool(F.relu(self.conv3(x)))  # After conv3 + pool
            self.flat_features = x.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 