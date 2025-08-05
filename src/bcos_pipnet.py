import torch
import torch.nn as nn
import torch.nn.functional as F
from bcos_features import bcos_resnet18_features, bcos_resnet50_features
import sys
import os

# Add PIPNet modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PIPNet'))
from pipnet.pipnet import NonNegLinear


class BcosPIPNet(nn.Module):
    """
    PIP-Net with B-cos backbone for self-supervised pre-training
    """
    def __init__(self, 
                 num_prototypes=512,
                 backbone='bcos_resnet18',
                 pretrained=False):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        
        # B-cos backbone for 6-channel input
        if backbone == 'bcos_resnet18':
            self.backbone = bcos_resnet18_features(pretrained=pretrained)
            backbone_out_channels = 512  # After global avgpool in backbone
        elif backbone == 'bcos_resnet50':
            self.backbone = bcos_resnet50_features(pretrained=pretrained)
            backbone_out_channels = 2048  # After global avgpool in backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Prototype projection from flattened backbone features
        # Since backbone now outputs flattened features, we need to project to prototypes
        self.prototype_projection = nn.Linear(backbone_out_channels, num_prototypes)
        
        # Non-negative activation for prototype similarity (like in PIP-Net)
        self.prototype_activation = nn.ReLU(inplace=True)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(num_prototypes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass for self-supervised pre-training
        
        Args:
            x: Input tensor of shape (batch_size, 6, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features=True: (proto_features, pooled_features, projected_features)
            Otherwise: projected_features
        """
        # Extract features with B-cos backbone (now returns flattened features)
        backbone_features = self.backbone(x)  # (batch_size, backbone_channels)
        
        # Project to prototype space
        proto_features = self.prototype_projection(backbone_features)  # (batch_size, num_prototypes)
        
        # Apply non-negative activation (prototype similarities should be positive)
        pooled_features = self.prototype_activation(proto_features)  # (batch_size, num_prototypes)
        
        # Project for contrastive learning
        projected_features = self.projection_head(pooled_features)  # (batch_size, 128)
        
        if return_features:
            return proto_features, pooled_features, projected_features
        else:
            return projected_features
    
    def get_prototype_activations(self, x):
        """
        Get prototype activations for analysis
        """
        with torch.no_grad():
            proto_features, pooled_features, _ = self.forward(x, return_features=True)
        return proto_features, pooled_features


class BcosPIPNetClassifier(nn.Module):
    """
    BcosPIPNet with classification head for fine-tuning
    """
    def __init__(self, 
                 pretrained_model,
                 num_classes,
                 freeze_backbone=True):
        super().__init__()
        
        # Copy pretrained components
        self.backbone = pretrained_model.backbone
        self.prototype_projection = pretrained_model.prototype_projection
        self.prototype_activation = pretrained_model.prototype_activation
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.prototype_projection.parameters():
                param.requires_grad = False
        
        # Classification layer (non-negative linear like in PIP-Net)
        self.classifier = NonNegLinear(pretrained_model.num_prototypes, num_classes, bias=True)
        
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, mean=1.0, std=0.1)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.)
    
    def forward(self, x, inference=False):
        """
        Forward pass for classification
        """
        # Extract features (now returns flattened features)
        backbone_features = self.backbone(x)
        proto_features = self.prototype_projection(backbone_features)
        pooled_features = self.prototype_activation(proto_features)
        
        # Apply threshold during inference (like in PIP-Net)
        if inference:
            pooled_features = torch.where(pooled_features < 0.1, 0., pooled_features)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        return proto_features, pooled_features, logits


def create_bcos_pipnet(num_prototypes=512, backbone='bcos_resnet18', pretrained=False):
    """
    Factory function to create BcosPIPNet model
    """
    return BcosPIPNet(
        num_prototypes=num_prototypes,
        backbone=backbone,
        pretrained=pretrained
    )


def create_bcos_pipnet_classifier(pretrained_path, num_classes, freeze_backbone=True):
    """
    Factory function to create BcosPIPNet classifier from pretrained model
    """
    # Load pretrained model
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Create base model
    pretrained_model = create_bcos_pipnet(
        num_prototypes=checkpoint.get('num_prototypes', 512),
        backbone=checkpoint.get('backbone', 'bcos_resnet18')
    )
    
    # Load weights
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier
    return BcosPIPNetClassifier(
        pretrained_model=pretrained_model,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )