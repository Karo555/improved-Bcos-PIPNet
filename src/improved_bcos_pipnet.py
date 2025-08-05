import torch
import torch.nn as nn
import torch.nn.functional as F
from bcos_features import bcos_resnet18_features, bcos_resnet50_features
import sys
import os
import numpy as np

# Add PIPNet modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PIPNet'))
from pipnet.pipnet import NonNegLinear


class ImprovedPrototypeLayer(nn.Module):
    """
    Improved prototype layer with better initialization and learning dynamics
    """
    def __init__(self, in_channels, num_prototypes, init_method='kmeans'):
        super().__init__()
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes
        self.init_method = init_method
        
        # Prototype transformation layer
        self.prototype_conv = nn.Conv2d(
            in_channels, num_prototypes, 
            kernel_size=1, stride=1, padding=0, bias=True
        )
        
        # Prototype activation layer (learnable activation function)
        self.activation = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(num_prototypes),
            nn.Sigmoid()  # Bound activations to [0, 1]
        )
        
        # Spatial attention mechanism for better localization
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_prototypes, num_prototypes // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_prototypes // 4, num_prototypes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization for prototype learning"""
        # Initialize prototype conv with Xavier uniform
        nn.init.xavier_uniform_(self.prototype_conv.weight)
        nn.init.constant_(self.prototype_conv.bias, 0.1)
        
        # Initialize spatial attention
        for m in self.spatial_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with improved prototype computation
        
        Args:
            x: Input features (batch_size, in_channels, H, W)
        
        Returns:
            prototype_features: Prototype activations (batch_size, num_prototypes, H, W)
        """
        # Generate prototype activations
        proto_raw = self.prototype_conv(x)
        
        # Apply learnable activation
        proto_activated = self.activation(proto_raw)
        
        # Apply spatial attention for better localization
        attention_weights = self.spatial_attention(proto_activated)
        proto_features = proto_activated * attention_weights
        
        return proto_features


class ImprovedBcosPIPNet(nn.Module):
    """
    Improved B-cos PIP-Net with better prototype learning
    """
    def __init__(self, 
                 num_prototypes=512,
                 backbone='bcos_resnet18',
                 pretrained=False,
                 prototype_init='improved',
                 dropout_rate=0.1):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.dropout_rate = dropout_rate
        
        # B-cos backbone for 6-channel input
        if backbone == 'bcos_resnet18':
            self.backbone = bcos_resnet18_features(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'bcos_resnet50':
            self.backbone = bcos_resnet50_features(pretrained=pretrained)
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Improved prototype layer
        self.prototype_layer = ImprovedPrototypeLayer(
            backbone_out_channels, 
            num_prototypes,
            init_method=prototype_init
        )
        
        # Global pooling with learnable aggregation
        self.global_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection head for contrastive learning (improved)
        self.projection_head = nn.Sequential(
            nn.Linear(num_prototypes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Initialize projection head
        self._initialize_projection_head()
    
    def _initialize_projection_head(self):
        """Initialize projection head weights"""
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass for improved self-supervised pre-training
        
        Args:
            x: Input tensor of shape (batch_size, 6, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features=True: (proto_features, pooled_features, projected_features)
            Otherwise: projected_features
        """
        # Extract features with B-cos backbone
        backbone_features = self.backbone(x)  # (batch_size, backbone_channels, H', W')
        
        # Generate improved prototype features
        proto_features = self.prototype_layer(backbone_features)  # (batch_size, num_prototypes, H', W')
        
        # Global pooling to get prototype presence scores
        pooled_features = self.global_pool(proto_features)  # (batch_size, num_prototypes, 1, 1)
        pooled_features = pooled_features.squeeze(-1).squeeze(-1)  # (batch_size, num_prototypes)
        
        # Apply dropout
        pooled_features = self.dropout(pooled_features)
        
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
    
    def freeze_backbone(self):
        """Freeze backbone for prototype-only learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end learning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ProgressiveTrainingStrategy:
    """
    Progressive training strategy for better prototype learning
    """
    def __init__(self, total_epochs=200, warmup_epochs=20, freeze_epochs=50):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_training_config(self):
        """Get training configuration based on current epoch"""
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: Warmup - Focus on prototype initialization
            return {
                'phase': 'warmup',
                'freeze_backbone': True,
                'learning_rate_multiplier': 0.1,
                'prototype_weight': 3.0,
                'contrastive_weight': 0.1,
                'align_weight': 0.5
            }
        elif self.current_epoch < self.freeze_epochs:
            # Phase 2: Prototype learning - Backbone frozen
            return {
                'phase': 'prototype_learning',
                'freeze_backbone': True,
                'learning_rate_multiplier': 1.0,
                'prototype_weight': 2.0,
                'contrastive_weight': 0.5,
                'align_weight': 1.0
            }
        else:
            # Phase 3: Joint training - Backbone unfrozen
            return {
                'phase': 'joint_training',
                'freeze_backbone': False,
                'learning_rate_multiplier': 1.0,
                'prototype_weight': 1.5,
                'contrastive_weight': 1.0,
                'align_weight': 1.0
            }


def create_improved_bcos_pipnet(num_prototypes=512, backbone='bcos_resnet18', 
                               pretrained=False, dropout_rate=0.1):
    """
    Factory function to create improved BcosPIPNet model
    """
    return ImprovedBcosPIPNet(
        num_prototypes=num_prototypes,
        backbone=backbone,
        pretrained=pretrained,
        prototype_init='improved',
        dropout_rate=dropout_rate
    )


class PrototypeMonitor:
    """
    Monitor prototype learning progress
    """
    def __init__(self, num_prototypes):
        self.num_prototypes = num_prototypes
        self.prototype_stats = []
    
    def update(self, model, data_loader, device, max_batches=10):
        """Update prototype statistics"""
        model.eval()
        activations = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= max_batches:
                    break
                    
                inputs = inputs.to(device)
                proto_features, pooled_features, _ = model(inputs, return_features=True)
                activations.append(pooled_features.cpu())
        
        if activations:
            all_activations = torch.cat(activations, dim=0)  # (total_samples, num_prototypes)
            
            stats = {
                'mean_activation': all_activations.mean(dim=0),  # Per prototype
                'std_activation': all_activations.std(dim=0),
                'active_ratio': (all_activations > 0.1).float().mean(dim=0),
                'max_activation': all_activations.max(dim=0)[0],
                'sparsity': (all_activations > 0.1).float().mean().item(),
                'diversity': self._compute_diversity(all_activations)
            }
            
            self.prototype_stats.append(stats)
            return stats
        
        return None
    
    def _compute_diversity(self, activations):
        """Compute prototype diversity measure"""
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(activations.t())
        
        # Remove diagonal and compute mean absolute correlation
        mask = torch.eye(self.num_prototypes, dtype=torch.bool)
        off_diag_corr = corr_matrix[~mask]
        
        # Diversity = 1 - mean_absolute_correlation
        diversity = 1.0 - torch.mean(torch.abs(off_diag_corr)).item()
        return diversity
    
    def print_summary(self):
        """Print summary of prototype learning"""
        if not self.prototype_stats:
            print("No prototype statistics available")
            return
        
        latest_stats = self.prototype_stats[-1]
        
        print("Prototype Learning Summary:")
        print(f"  Sparsity (active ratio): {latest_stats['sparsity']:.3f}")
        print(f"  Diversity score: {latest_stats['diversity']:.3f}")
        print(f"  Mean activation: {latest_stats['mean_activation'].mean().item():.3f}")
        print(f"  Active prototypes: {(latest_stats['active_ratio'] > 0.1).sum().item()}/{self.num_prototypes}")
        
        # Check for common issues
        if latest_stats['sparsity'] < 0.1:
            print("  ⚠️  Low sparsity - prototypes may not be learning distinctive patterns")
        if latest_stats['diversity'] < 0.5:
            print("  ⚠️  Low diversity - prototypes may be too similar")
        if (latest_stats['active_ratio'] > 0.1).sum().item() < self.num_prototypes * 0.3:
            print("  ⚠️  Many inactive prototypes - consider reducing number or adjusting loss")
        
        print(f"  ✓ Prototype learning appears {'healthy' if latest_stats['diversity'] > 0.5 and latest_stats['sparsity'] > 0.1 else 'problematic'}")


def debug_prototype_learning(model, data_loader, device, epoch=0):
    """
    Debug prototype learning issues
    """
    model.eval()
    print(f"\n=== Debugging Prototype Learning (Epoch {epoch}) ===")
    
    # Get a batch of data
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        # Forward pass
        proto_features, pooled_features, projected = model(inputs, return_features=True)
        
        print(f"Shapes:")
        print(f"  Input: {inputs.shape}")
        print(f"  Prototype features: {proto_features.shape}")
        print(f"  Pooled features: {pooled_features.shape}")
        print(f"  Projected features: {projected.shape}")
        
        print(f"\nActivation Statistics:")
        print(f"  Prototype features - min: {proto_features.min().item():.4f}, max: {proto_features.max().item():.4f}, mean: {proto_features.mean().item():.4f}")
        print(f"  Pooled features - min: {pooled_features.min().item():.4f}, max: {pooled_features.max().item():.4f}, mean: {pooled_features.mean().item():.4f}")
        
        # Check for dead prototypes
        active_prototypes = (pooled_features > 0.01).any(dim=0).sum().item()
        print(f"  Active prototypes (>0.01): {active_prototypes}/{model.num_prototypes}")
        
        # Check for saturated prototypes
        saturated_prototypes = (pooled_features > 0.99).any(dim=0).sum().item()
        print(f"  Saturated prototypes (>0.99): {saturated_prototypes}/{model.num_prototypes}")
        
        # Prototype diversity
        proto_corr = torch.corrcoef(pooled_features.t())
        mask = torch.eye(model.num_prototypes, dtype=torch.bool, device=pooled_features.device)
        mean_corr = proto_corr[~mask].abs().mean().item()
        print(f"  Mean prototype correlation: {mean_corr:.4f}")
        
        print(f"\nRecommendations:")
        if active_prototypes < model.num_prototypes * 0.5:
            print("  - Consider reducing prototype learning rate or increasing prototype loss weight")
        if mean_corr > 0.8:
            print("  - Prototypes are too similar - increase diversity regularization")
        if saturated_prototypes > model.num_prototypes * 0.1:
            print("  - Some prototypes are saturated - consider gradient clipping or lower learning rate")
        
        print("=" * 50)