import torch
import torch.nn as nn
import sys
import os

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d


class BcosResNet18Features(nn.Module):
    """
    ResNet18-like backbone with BcosConv2d layers for 6-channel input
    """
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Initial conv layer for 6-channel input
        self.conv1 = BcosConv2d(6, 64, kernel_size=7, stride=2, padding=3, b=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks with BcosConv2d
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
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
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class BcosBasicBlock(nn.Module):
    """
    Basic ResNet block with BcosConv2d
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = BcosConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, b=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = BcosConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, b=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BcosConv2d(in_channels, out_channels, kernel_size=1, stride=stride, b=2),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu2(out)
        
        return out


class BcosResNet50Features(nn.Module):
    """
    ResNet50-like backbone with BcosConv2d layers for 6-channel input
    """
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Initial conv layer for 6-channel input
        self.conv1 = BcosConv2d(6, 64, kernel_size=7, stride=2, padding=3, b=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet bottleneck blocks
        self.layer1 = self._make_layer(64, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, 3, stride=2)
        
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
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class BcosBottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with BcosConv2d
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = BcosConv2d(in_channels, mid_channels, kernel_size=1, stride=1, b=2)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = BcosConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, b=2)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = BcosConv2d(mid_channels, out_channels, kernel_size=1, stride=1, b=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BcosConv2d(in_channels, out_channels, kernel_size=1, stride=stride, b=2),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu3(out)
        
        return out


def bcos_resnet18_features(pretrained=False):
    """Construct B-cos ResNet18 feature extractor"""
    return BcosResNet18Features(pretrained=pretrained)


def bcos_resnet50_features(pretrained=False):
    """Construct B-cos ResNet50 feature extractor"""
    return BcosResNet50Features(pretrained=pretrained)