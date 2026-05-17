import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNet(BaseModel):
    def __init__(self, num_classes=10, **kwargs):
        super(MobileNet, self).__init__()
        
        # Get width multiplier from kwargs with default value
        width_multiplier = kwargs.get('width_multiplier', 1.0)
        
        # Apply width multiplier to all channel dimensions
        input_channel = int(32 * width_multiplier)
        self.last_channel = int(1024 * width_multiplier)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # MobileNet blocks
        self.conv2 = DepthwiseSeparableConv(input_channel, int(64 * width_multiplier), stride=1)
        self.conv3 = DepthwiseSeparableConv(int(64 * width_multiplier), int(128 * width_multiplier), stride=2)
        self.conv4 = DepthwiseSeparableConv(int(128 * width_multiplier), int(128 * width_multiplier), stride=1)
        self.conv5 = DepthwiseSeparableConv(int(128 * width_multiplier), int(256 * width_multiplier), stride=2)
        self.conv6 = DepthwiseSeparableConv(int(256 * width_multiplier), int(256 * width_multiplier), stride=1)
        self.conv7 = DepthwiseSeparableConv(int(256 * width_multiplier), int(512 * width_multiplier), stride=2)
        
        # Additional layers for MNIST (smaller images)
        self.conv8 = DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1)
        self.conv9 = DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1)
        self.conv10 = DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1)
        self.conv11 = DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1)
        self.conv12 = DepthwiseSeparableConv(int(512 * width_multiplier), int(1024 * width_multiplier), stride=2)
        self.conv13 = DepthwiseSeparableConv(int(1024 * width_multiplier), self.last_channel, stride=1)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.last_channel, num_classes)
        
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
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # MobileNet blocks
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 