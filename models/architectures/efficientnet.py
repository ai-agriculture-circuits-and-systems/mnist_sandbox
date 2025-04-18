import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBnAct(nn.Module):
    """Convolution block with batch normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 groups=1, bias=False, act_fn=Swish()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConv(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expansion=6, reduction=4, drop_rate=0.):
        super().__init__()
        expanded_channels = in_channels * expansion
        padding = kernel_size // 2
        
        # Expansion phase
        self.expand = nn.Sequential()
        if expansion != 1:
            self.expand = ConvBnAct(in_channels, expanded_channels, kernel_size=1)
        
        # Depthwise convolution phase
        self.depthwise = ConvBnAct(expanded_channels, expanded_channels, 
                                  kernel_size=kernel_size, stride=stride, 
                                  padding=padding, groups=expanded_channels)
        
        # Squeeze and Excitation phase
        self.se = SqueezeExcite(expanded_channels, reduction)
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Sequential()
        
        # Dropout
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        skip = self.skip(x)
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        x = self.dropout(x)
        
        x = x + skip
        return x

class EfficientNet(BaseModel):
    """EfficientNet model for MNIST classification."""
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, 
                 dropout_rate=0.2, reduction=4):
        super().__init__()
        
        # Stem
        channels = int(32 * width_mult)
        self.stem = ConvBnAct(1, channels, kernel_size=3, stride=2, padding=1)
        
        # Body
        self.blocks = nn.ModuleList([
            # Stage 1
            MBConv(channels, int(16 * width_mult), kernel_size=3, stride=1, 
                  expansion=1, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 2
            MBConv(int(16 * width_mult), int(24 * width_mult), kernel_size=3, stride=2, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(24 * width_mult), int(24 * width_mult), kernel_size=3, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 3
            MBConv(int(24 * width_mult), int(40 * width_mult), kernel_size=5, stride=2, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(40 * width_mult), int(40 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 4
            MBConv(int(40 * width_mult), int(80 * width_mult), kernel_size=3, stride=2, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(80 * width_mult), int(80 * width_mult), kernel_size=3, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(80 * width_mult), int(80 * width_mult), kernel_size=3, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 5
            MBConv(int(80 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(112 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(112 * width_mult), int(112 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 6
            MBConv(int(112 * width_mult), int(192 * width_mult), kernel_size=5, stride=2, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(192 * width_mult), int(192 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(192 * width_mult), int(192 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            MBConv(int(192 * width_mult), int(192 * width_mult), kernel_size=5, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
            
            # Stage 7
            MBConv(int(192 * width_mult), int(320 * width_mult), kernel_size=3, stride=1, 
                  expansion=6, reduction=reduction, drop_rate=dropout_rate),
        ])
        
        # Head
        self.head = nn.Sequential(
            ConvBnAct(int(320 * width_mult), int(1280 * width_mult), kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(1280 * width_mult), num_classes)
        )
        
        # Initialize weights
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
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x 