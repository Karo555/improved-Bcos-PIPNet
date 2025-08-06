import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from bcos_features import (
    bcos_simple_features, bcos_medium_features, bcos_large_features,
    bcos_resnet18_features, bcos_resnet50_features  # Keep for backward compatibility
)
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
                 backbone='bcos_simple',
                 pretrained=False):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        
        # B-cos backbone for 6-channel input
        if backbone == 'bcos_simple':
            self.backbone = bcos_simple_features(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'bcos_medium':
            self.backbone = bcos_medium_features(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'bcos_large':
            self.backbone = bcos_large_features(pretrained=pretrained)
            backbone_out_channels = 1024
        elif backbone == 'bcos_resnet18':  # Backward compatibility - now uses simple CNN
            self.backbone = bcos_resnet18_features(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'bcos_resnet50':  # Backward compatibility - now uses medium CNN
            self.backbone = bcos_resnet50_features(pretrained=pretrained)
            backbone_out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Supported: bcos_simple, bcos_medium, bcos_large, bcos_resnet18, bcos_resnet50")
        
        # Prototype projection from flattened backbone features
        # Since backbone now outputs flattened features, we need to project to prototypes
        self.prototype_projection = nn.Linear(backbone_out_channels, num_prototypes)
        
        # Non-negative activation for prototype similarity - use absolute value instead of ReLU
        # This maintains B-cos interpretability while ensuring positive prototype activations
        
        # Projection head for contrastive learning - remove ReLU to maintain B-cos principles
        self.projection_head = nn.Sequential(
            nn.Linear(num_prototypes, 256),
            nn.Linear(256, 128)  # Removed ReLU activation
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
        
        # Apply absolute value for non-negative activation (B-cos compatible)
        pooled_features = torch.abs(proto_features)  # (batch_size, num_prototypes)
        
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
        Forward pass for classification with PIPNet-style sparsity
        """
        # Extract features (now returns flattened features)
        backbone_features = self.backbone(x)
        proto_features = self.prototype_projection(backbone_features)
        pooled_features = torch.abs(proto_features)  # Use abs instead of ReLU
        
        # Apply sparsity threshold during inference (PIPNet-style)
        if inference:
            pooled_features = torch.where(pooled_features < 0.1, 0., pooled_features)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        return proto_features, pooled_features, logits
    
    def apply_weight_clamping(self, threshold=1e-3):
        """
        Apply PIPNet-style aggressive weight clamping to classification layer
        Sets weights below threshold to zero to enforce sparsity
        """
        with torch.no_grad():
            self.classifier.weight.copy_(
                torch.clamp(self.classifier.weight.data - threshold, min=0.)
            )
    
    def get_sparsity_metrics(self):
        """
        Get sparsity metrics for monitoring prototype usage
        
        Returns:
            dict: Sparsity metrics including sparsity ratio and active prototypes
        """
        with torch.no_grad():
            # Classification weight sparsity
            total_weights = torch.numel(self.classifier.weight)
            active_weights = torch.count_nonzero(
                torch.relu(self.classifier.weight - 1e-3)
            ).item()
            sparsity_ratio = (total_weights - active_weights) / total_weights
            
            # Count prototypes with any non-zero weights
            active_prototypes = torch.sum(
                torch.any(self.classifier.weight > 1e-3, dim=0)
            ).item()
            
            return {
                'sparsity_ratio': sparsity_ratio,
                'active_prototypes': active_prototypes,
                'total_prototypes': self.classifier.in_features,
                'total_weights': total_weights,
                'active_weights': active_weights
            }
    
    def prune_unused_prototypes(self, data_loader, device, activation_threshold=0.1):
        """
        Post-training pruning: zero out prototypes that are never significantly activated
        
        Args:
            data_loader: DataLoader to evaluate prototype usage
            device: Device to run evaluation on
            activation_threshold: Minimum activation to consider a prototype as "used"
            
        Returns:
            dict: Pruning statistics
        """
        self.eval()
        prototype_max_activations = torch.zeros(self.classifier.in_features, device=device)
        
        print(f"Evaluating prototype usage on {len(data_loader.dataset)} samples...")
        
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc="Computing prototype activations"):
                inputs = inputs.to(device)
                
                # Get prototype activations
                _, pooled_features, _ = self.forward(inputs, inference=True)
                
                # Track maximum activation for each prototype
                batch_max = torch.max(pooled_features, dim=0)[0]
                prototype_max_activations = torch.maximum(prototype_max_activations, batch_max)
        
        # Find unused prototypes
        unused_prototypes = prototype_max_activations < activation_threshold
        num_unused = unused_prototypes.sum().item()
        
        print(f"Found {num_unused}/{self.classifier.in_features} unused prototypes")
        
        if num_unused > 0:
            # Zero out unused prototypes in classification layer
            with torch.no_grad():
                self.classifier.weight[:, unused_prototypes] = 0.0
            print(f"Pruned {num_unused} unused prototypes from classification layer")
        
        return {
            'unused_prototypes': num_unused,
            'total_prototypes': self.classifier.in_features,
            'pruning_ratio': num_unused / self.classifier.in_features,
            'unused_prototype_indices': torch.where(unused_prototypes)[0].tolist(),
            'prototype_max_activations': prototype_max_activations.cpu().tolist()
        }


def create_bcos_pipnet(num_prototypes=512, backbone='bcos_simple', pretrained=False):
    """
    Factory function to create BcosPIPNet model
    
    Args:
        num_prototypes: Number of prototypes to learn
        backbone: Backbone architecture ('bcos_simple', 'bcos_medium', 'bcos_large')
        pretrained: Whether to load pretrained weights (always False for now)
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
    
    # Create base model - default to simple if backbone not specified
    pretrained_model = create_bcos_pipnet(
        num_prototypes=checkpoint.get('num_prototypes', 512),
        backbone=checkpoint.get('backbone', 'bcos_simple')
    )
    
    # Load weights
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier
    return BcosPIPNetClassifier(
        pretrained_model=pretrained_model,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )