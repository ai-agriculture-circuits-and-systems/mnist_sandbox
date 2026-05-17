import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
import torch

class VGG(BaseModel):
    def __init__(self, num_classes=10, cfg='A', input_size=224):
        super(VGG, self).__init__()
        
        # VGG configurations
        cfg_dict = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        # Get the configuration
        self.cfg = cfg_dict[cfg]
        
        # Build the features
        self.features = self._make_layers(self.cfg)
        
        # Calculate the size of the flattened features
        self._initialize_weights()
        
        # Calculate the size of the flattened features based on input size
        with torch.no_grad():
            # Create a dummy input to calculate the output size
            dummy_input = torch.zeros(1, 1, input_size, input_size)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # MNIST has 1 channel
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = v
                
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 