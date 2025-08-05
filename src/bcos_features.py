import torch
import torch.nn as nn
import sys
import os

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d


class BcosResNet18Features(nn.Module):
    """
    B-cos ResNet18-like backbone for 6-channel input
    NO ReLU, NO BatchNorm, NO MaxPooling - only B-cos convolutions with built-in MaxOut
    """
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Initial B-cos conv layer for 6-channel input with built-in MaxOut
        self.conv1 = BcosConv2d(6, 64, kernel_size=7, stride=2, padding=3, max_out=2, b=2)
        
        # B-cos ResNet blocks - no BatchNorm, no ReLU, no MaxPool
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling instead of MaxPool
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block might have stride != 1
        layers.append(BcosBasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BcosBasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize B-cos weights properly"""
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                # B-cos specific initialization - linear activation
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='linear')
    
    def forward(self, x):
        # Pure B-cos forward - no ReLU, no BatchNorm, no MaxPool
        x = self.conv1(x)  # Built-in MaxOut in BcosConv2d
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


class BcosBasicBlock(nn.Module):
    """
    Basic ResNet block with BcosConv2d
    NO ReLU, NO BatchNorm - uses only B-cos convolutions with built-in MaxOut
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = BcosConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, max_out=2, b=2)
        self.conv2 = BcosConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, max_out=2, b=2)
        
        # Shortcut connection - also B-cos, no BatchNorm
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BcosConv2d(in_channels, out_channels, kernel_size=1, stride=stride, max_out=2, b=2)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)  # MaxOut built into BcosConv2d
        out = self.conv2(out)  # MaxOut built into BcosConv2d
        
        # Add residual (B-cos networks maintain this)
        out += residual
        
        return out


class BcosResNet50Features(nn.Module):
    """
    B-cos ResNet50-like backbone for 6-channel input
    NO ReLU, NO BatchNorm, NO MaxPooling - only B-cos convolutions with built-in MaxOut
    """
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Initial conv layer for 6-channel input
        self.conv1 = BcosConv2d(6, 64, kernel_size=7, stride=2, padding=3, max_out=2, b=2)
        
        # ResNet bottleneck blocks - no BatchNorm, no ReLU, no MaxPool
        self.layer1 = self._make_layer(64, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, 3, stride=2)
        
        # Global average pooling instead of MaxPool
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block might have stride != 1
        layers.append(BcosBottleneckBlock(in_channels, mid_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BcosBottleneckBlock(out_channels, mid_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='linear')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


class BcosBottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with BcosConv2d
    NO ReLU, NO BatchNorm - uses only B-cos convolutions with built-in MaxOut
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = BcosConv2d(in_channels, mid_channels, kernel_size=1, stride=1, max_out=2, b=2)
        self.conv2 = BcosConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, max_out=2, b=2)
        self.conv3 = BcosConv2d(mid_channels, out_channels, kernel_size=1, stride=1, max_out=2, b=2)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BcosConv2d(in_channels, out_channels, kernel_size=1, stride=stride, max_out=2, b=2)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)  # MaxOut built into BcosConv2d
        out = self.conv2(out)  # MaxOut built into BcosConv2d
        out = self.conv3(out)  # MaxOut built into BcosConv2d
        
        out += residual
        
        return out


def bcos_resnet18_features(pretrained=False):
    """Construct B-cos ResNet18 feature extractor"""
    return BcosResNet18Features(pretrained=pretrained)


def bcos_resnet50_features(pretrained=False):
    """Construct B-cos ResNet50 feature extractor"""
    return BcosResNet50Features(pretrained=pretrained)