import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class SeparableConv2d(nn.Module):
    """Separable 2D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EntryFlow(nn.Module):
    """Entry flow of Xception."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeparableConv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        
        self.block3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 728, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.skip3 = nn.Sequential(
            nn.Conv2d(256, 728, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        skip = self.skip1(x)
        x = self.block1(x)
        x = x + skip
        
        skip = self.skip2(x)
        x = self.block2(x)
        x = x + skip
        
        skip = self.skip3(x)
        x = self.block3(x)
        x = x + skip
        
        return x

class MiddleFlow(nn.Module):
    """Middle flow of Xception."""
    def __init__(self, num_blocks=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=True),
                SeparableConv2d(728, 728, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(728),
                nn.ReLU(inplace=True),
                SeparableConv2d(728, 728, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(728),
                nn.ReLU(inplace=True),
                SeparableConv2d(728, 728, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(728)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class ExitFlow(nn.Module):
    """Exit flow of Xception."""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.skip1 = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )
        
        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(1024, 1536, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip = self.skip1(x)
        x = self.block1(x)
        x = x + skip
        x = self.block2(x)
        return x

class Xception(BaseModel):
    """Xception model for MNIST classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
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
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x 