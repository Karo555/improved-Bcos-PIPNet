import torch
import torch.nn as nn
import sys
import os

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d
from modules.utils import MyAdaptiveAvgPool2d


class SimpleBcosFeatures(nn.Module):
    """
    Simple B-cos CNN backbone following B-cos CIFAR10 experiments
    Sequential B-cos convolutions with strided convs for downsampling
    NO ResNet structure, NO ReLU, NO BatchNorm, NO MaxPooling
    """
    def __init__(self, input_channels=6, pretrained=False):
        super().__init__()
        
        # Configuration based on B-cos CIFAR10 experiments
        # [channels, kernel_size, stride, padding]
        self.config = [
            [64, 3, 1, 1],    # Initial conv
            [128, 3, 2, 1],   # Downsample
            [128, 3, 1, 1],   # Feature extraction
            [256, 3, 2, 1],   # Downsample
            [256, 3, 1, 1],   # Feature extraction
            [512, 3, 2, 1],   # Downsample
            [512, 3, 1, 1],   # Feature extraction
        ]
        
        # Build sequential B-cos layers
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride, padding in self.config:
            layers.append(
                BcosConv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    max_out=2,  # Built-in MaxOut
                    b=2,        # B-cos parameter
                    scale_fact=100  # Scale factor like in B-cos experiments
                )
            )
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling (B-cos style)
        self.global_avgpool = MyAdaptiveAvgPool2d((1, 1))
        
        self.output_channels = 512
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize B-cos weights following their experiments"""
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                # B-cos uses linear nonlinearity for initialization
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='linear')
    
    def forward(self, x):
        """Forward pass through sequential B-cos layers"""
        x = self.features(x)  # Sequential B-cos convolutions
        x = self.global_avgpool(x)  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten
        return x


class MediumBcosFeatures(nn.Module):
    """
    Medium-sized B-cos CNN backbone
    More layers than simple version for better feature extraction
    """
    def __init__(self, input_channels=6, pretrained=False):
        super().__init__()
        
        # Larger configuration for more complex datasets
        self.config = [
            [64, 3, 1, 1],     # Initial conv
            [64, 3, 1, 1],     # Feature extraction
            [128, 3, 2, 1],    # Downsample
            [128, 3, 1, 1],    # Feature extraction
            [256, 3, 2, 1],    # Downsample
            [256, 3, 1, 1],    # Feature extraction
            [256, 3, 1, 1],    # Feature extraction
            [512, 3, 2, 1],    # Downsample
            [512, 3, 1, 1],    # Feature extraction
            [512, 3, 1, 1],    # Feature extraction
        ]
        
        # Build sequential B-cos layers
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride, padding in self.config:
            layers.append(
                BcosConv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    max_out=2,
                    b=2,
                    scale_fact=100
                )
            )
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_avgpool = MyAdaptiveAvgPool2d((1, 1))
        
        self.output_channels = 512
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='linear')
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class LargeBcosFeatures(nn.Module):
    """
    Large B-cos CNN backbone for complex datasets like CUB-200
    """
    def __init__(self, input_channels=6, pretrained=False):
        super().__init__()
        
        # Large configuration
        self.config = [
            [64, 3, 1, 1],     # Initial conv
            [64, 3, 1, 1],     # Feature extraction
            [128, 3, 2, 1],    # Downsample
            [128, 3, 1, 1],    # Feature extraction
            [128, 3, 1, 1],    # Feature extraction
            [256, 3, 2, 1],    # Downsample
            [256, 3, 1, 1],    # Feature extraction
            [256, 3, 1, 1],    # Feature extraction
            [512, 3, 2, 1],    # Downsample
            [512, 3, 1, 1],    # Feature extraction
            [512, 3, 1, 1],    # Feature extraction
            [1024, 3, 2, 1],   # Downsample
            [1024, 3, 1, 1],   # Feature extraction
        ]
        
        # Build sequential B-cos layers
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride, padding in self.config:
            layers.append(
                BcosConv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    max_out=2,
                    b=2,
                    scale_fact=100
                )
            )
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_avgpool = MyAdaptiveAvgPool2d((1, 1))
        
        self.output_channels = 1024
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BcosConv2d):
                nn.init.kaiming_normal_(m.linear.weight, mode='fan_out', nonlinearity='linear')
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def bcos_simple_features(pretrained=False):
    """Construct simple B-cos CNN feature extractor (512 output channels)"""
    return SimpleBcosFeatures(input_channels=6, pretrained=pretrained)


def bcos_medium_features(pretrained=False):
    """Construct medium B-cos CNN feature extractor (512 output channels)"""
    return MediumBcosFeatures(input_channels=6, pretrained=pretrained)


def bcos_large_features(pretrained=False):
    """Construct large B-cos CNN feature extractor (1024 output channels)"""
    return LargeBcosFeatures(input_channels=6, pretrained=pretrained)


# Keep compatibility but these now use simple B-cos CNNs instead of ResNet
def bcos_resnet18_features(pretrained=False):
    """Construct simple B-cos CNN feature extractor (replaces ResNet18)"""
    return SimpleBcosFeatures(input_channels=6, pretrained=pretrained)


def bcos_resnet50_features(pretrained=False):
    """Construct medium B-cos CNN feature extractor (replaces ResNet50)"""
    return MediumBcosFeatures(input_channels=6, pretrained=pretrained)