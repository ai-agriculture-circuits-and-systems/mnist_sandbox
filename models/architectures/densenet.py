import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return self.pool(out)

class DenseNet(BaseModel):
    def __init__(self, num_classes=10, growth_rate=12, block_config=(3, 6, 12, 8), drop_rate=0.0):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers, drop_rate)
            self.dense_blocks.append(block)
            num_channels += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2
        
        # Final batch norm
        self.bn_final = nn.BatchNorm2d(num_channels)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        for i, dense_block in enumerate(self.dense_blocks):
            out = dense_block(out)
            if i != len(self.dense_blocks) - 1:
                out = self.transitions[i](out)
        
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out 