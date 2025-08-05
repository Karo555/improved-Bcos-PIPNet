import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d


class ImprovedAlignLoss(nn.Module):
    """
    Improved Alignment loss for B-cos Networks
    Better implementation that encourages meaningful weight diversity
    """
    def __init__(self, weight=1.0, diversity_weight=0.5):
        super().__init__()
        self.weight = weight
        self.diversity_weight = diversity_weight
    
    def forward(self, model):
        """
        Compute improved alignment loss across all BcosConv2d layers
        """
        align_loss = 0.0
        diversity_loss = 0.0
        num_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, BcosConv2d):
                # Get the weight tensor
                weight = module.linear.weight  # Shape: (out_channels, in_channels*kernel_size^2)
                
                # 1. Encourage unit norm (already done by NormedConv2d but reinforce)
                weight_norms = torch.norm(weight, dim=1, keepdim=True)
                norm_penalty = torch.mean((weight_norms - 1.0) ** 2)
                
                # 2. Encourage diversity between filters
                weight_flat = weight.view(weight.size(0), -1)
                weight_norm = F.normalize(weight_flat, dim=1)
                
                # Compute pairwise cosine similarities
                similarity_matrix = torch.mm(weight_norm, weight_norm.t())
                
                # Remove diagonal and encourage low off-diagonal similarities
                mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
                off_diagonal = similarity_matrix * (1 - mask)
                
                # Penalize high similarities (encourage diversity)
                diversity_penalty = torch.sum(off_diagonal ** 2)
                
                # 3. Encourage balanced filter magnitudes
                magnitude_std = torch.std(torch.norm(weight, dim=1))
                magnitude_penalty = magnitude_std  # Penalize high variance in magnitudes
                
                align_loss += norm_penalty + magnitude_penalty
                diversity_loss += diversity_penalty
                num_layers += 1
        
        if num_layers > 0:
            align_loss = align_loss / num_layers
            diversity_loss = diversity_loss / num_layers
        
        total_loss = self.weight * (align_loss + self.diversity_weight * diversity_loss)
        return total_loss


class ImprovedPrototypeLoss(nn.Module):
    """
    Improved prototype learning loss that replaces the problematic TanhLoss
    """
    def __init__(self, weight=1.0, sparsity_weight=0.1, entropy_weight=0.1, concentration_weight=0.5):
        super().__init__()
        self.weight = weight
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.concentration_weight = concentration_weight
    
    def forward(self, proto_features, pooled_features=None):
        """
        Comprehensive prototype learning loss
        
        Args:
            proto_features: Prototype feature maps (batch_size, num_prototypes, H, W)
            pooled_features: Global pooled features (batch_size, num_prototypes)
        """
        batch_size, num_prototypes, H, W = proto_features.shape
        
        # 1. Sparsity Loss: Encourage few but strong activations
        # Use L1 penalty on pooled features if available
        if pooled_features is not None:
            sparsity_loss = torch.mean(torch.abs(pooled_features))
        else:
            # Alternative: encourage spatial sparsity
            spatial_max = torch.max(proto_features.view(batch_size, num_prototypes, -1), dim=2)[0]
            sparsity_loss = torch.mean(torch.abs(spatial_max))
        
        # 2. Concentration Loss: Encourage localized activations
        # Compute spatial variance for each prototype
        proto_flat = proto_features.view(batch_size, num_prototypes, H * W)
        proto_mean = torch.mean(proto_flat, dim=2, keepdim=True)
        proto_var = torch.mean((proto_flat - proto_mean) ** 2, dim=2)
        concentration_loss = -torch.mean(proto_var)  # Negative because we want high variance (concentration)
        
        # 3. Diversity Loss: Encourage different prototypes to activate differently
        if pooled_features is not None:
            # Compute correlation between prototypes
            proto_norm = F.normalize(pooled_features, dim=0)  # Normalize across batch
            correlation_matrix = torch.mm(proto_norm.t(), proto_norm)
            
            # Penalize high correlations (encourage diversity)
            mask = torch.eye(num_prototypes, device=correlation_matrix.device)
            off_diagonal = correlation_matrix * (1 - mask)
            diversity_loss = torch.sum(off_diagonal ** 2)
        else:
            diversity_loss = torch.tensor(0.0, device=proto_features.device)
        
        # 4. Entropy Loss: Encourage balanced prototype usage
        if pooled_features is not None:
            # Compute prototype usage distribution
            proto_usage = torch.mean(torch.abs(pooled_features), dim=0)  # Mean usage per prototype
            proto_usage = proto_usage / (torch.sum(proto_usage) + 1e-8)  # Normalize
            
            # Compute entropy (higher is better for balanced usage)
            entropy = -torch.sum(proto_usage * torch.log(proto_usage + 1e-8))
            max_entropy = np.log(num_prototypes)
            entropy_loss = max_entropy - entropy  # Minimize negative entropy
        else:
            entropy_loss = torch.tensor(0.0, device=proto_features.device)
        
        # Combine losses
        total_loss = (
            self.sparsity_weight * sparsity_loss +
            self.concentration_weight * concentration_loss +
            self.entropy_weight * entropy_loss +
            0.1 * diversity_loss  # Small weight for diversity
        )
        
        return self.weight * total_loss, {
            'sparsity_loss': sparsity_loss,
            'concentration_loss': concentration_loss,
            'entropy_loss': entropy_loss,
            'diversity_loss': diversity_loss
        }


class ContrastivePrototypeLoss(nn.Module):
    """
    Contrastive loss specifically designed for prototype learning
    """
    def __init__(self, temperature=0.1, weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, proto_features1, proto_features2, pooled1=None, pooled2=None):
        """
        Contrastive loss between prototype features from two augmented views
        """
        batch_size = proto_features1.size(0)
        
        # Use pooled features if available, otherwise global average pool
        if pooled1 is None:
            pooled1 = F.adaptive_avg_pool2d(proto_features1, (1, 1)).squeeze()
        if pooled2 is None:
            pooled2 = F.adaptive_avg_pool2d(proto_features2, (1, 1)).squeeze()
        
        # Ensure 2D shape
        if pooled1.dim() == 1:
            pooled1 = pooled1.unsqueeze(0)
        if pooled2.dim() == 1:
            pooled2 = pooled2.unsqueeze(0)
        
        # Normalize features
        pooled1 = F.normalize(pooled1, dim=1)
        pooled2 = F.normalize(pooled2, dim=1)
        
        # Compute similarity matrix
        # Positive pairs: same image, different augmentations
        pos_sim = torch.sum(pooled1 * pooled2, dim=1) / self.temperature  # (batch_size,)
        
        # Negative pairs: different images
        all_features = torch.cat([pooled1, pooled2], dim=0)  # (2*batch_size, num_prototypes)
        sim_matrix = torch.mm(all_features, all_features.t()) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create mask for positive pairs
        batch_indices = torch.arange(batch_size, device=pooled1.device)
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=pooled1.device, dtype=torch.bool)
        pos_mask[batch_indices, batch_indices + batch_size] = True
        pos_mask[batch_indices + batch_size, batch_indices] = True
        
        # Remove self-similarities
        self_mask = torch.eye(2 * batch_size, device=pooled1.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))
        
        # InfoNCE loss
        pos_sim_full = sim_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = sim_matrix[~pos_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim_full, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=pooled1.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return self.weight * contrastive_loss


class ImprovedCombinedPretrainingLoss(nn.Module):
    """
    Improved combined loss for self-supervised pre-training
    """
    def __init__(self, 
                 align_weight=1.0,
                 prototype_weight=2.0,  # Increased weight for prototype learning
                 contrastive_weight=0.5,
                 consistency_weight=0.3):
        super().__init__()
        
        self.align_loss = ImprovedAlignLoss(weight=align_weight)
        self.prototype_loss = ImprovedPrototypeLoss(weight=prototype_weight)
        self.contrastive_loss = ContrastivePrototypeLoss(weight=contrastive_weight)
        self.consistency_weight = consistency_weight
    
    def forward(self, model, proto_features1, proto_features2, pooled1=None, pooled2=None):
        """
        Compute comprehensive pre-training loss
        """
        # Alignment loss from B-cos layers
        la = self.align_loss(model)
        
        # Prototype learning losses
        lp1, proto_details1 = self.prototype_loss(proto_features1, pooled1)
        lp2, proto_details2 = self.prototype_loss(proto_features2, pooled2)
        lp = (lp1 + lp2) / 2.0
        
        # Contrastive loss for consistency
        lc = self.contrastive_loss(proto_features1, proto_features2, pooled1, pooled2)
        
        # Consistency loss: encourage similar prototype activations for same image
        if pooled1 is not None and pooled2 is not None:
            consistency_loss = F.mse_loss(pooled1, pooled2)
        else:
            # Alternative: use global average pooling
            pool1 = F.adaptive_avg_pool2d(proto_features1, (1, 1)).squeeze()
            pool2 = F.adaptive_avg_pool2d(proto_features2, (1, 1)).squeeze()
            consistency_loss = F.mse_loss(pool1, pool2)
        
        consistency_loss = self.consistency_weight * consistency_loss
        
        total_loss = la + lp + lc + consistency_loss
        
        return {
            'total_loss': total_loss,
            'align_loss': la,
            'prototype_loss': lp,
            'contrastive_loss': lc,
            'consistency_loss': consistency_loss,
            'prototype_details': {
                'sparsity_loss': (proto_details1['sparsity_loss'] + proto_details2['sparsity_loss']) / 2,
                'concentration_loss': (proto_details1['concentration_loss'] + proto_details2['concentration_loss']) / 2,
                'entropy_loss': (proto_details1['entropy_loss'] + proto_details2['entropy_loss']) / 2,
                'diversity_loss': (proto_details1['diversity_loss'] + proto_details2['diversity_loss']) / 2,
            }
        }


class AdaptivePrototypeLoss(nn.Module):
    """
    Adaptive prototype loss that adjusts based on training progress
    """
    def __init__(self, 
                 initial_sparsity_weight=0.1,
                 initial_concentration_weight=0.5,
                 warmup_epochs=10,
                 total_epochs=200):
        super().__init__()
        self.initial_sparsity_weight = initial_sparsity_weight
        self.initial_concentration_weight = initial_concentration_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_adaptive_weights(self):
        """Adjust loss weights based on training progress"""
        progress = self.current_epoch / self.total_epochs
        
        # Early training: focus on concentration (localization)
        # Later training: focus on sparsity (selectivity)
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: emphasize concentration
            concentration_weight = self.initial_concentration_weight * 2.0
            sparsity_weight = self.initial_sparsity_weight * 0.5
        else:
            # Progressive shift from concentration to sparsity
            concentration_weight = self.initial_concentration_weight * (1.0 - progress * 0.5)
            sparsity_weight = self.initial_sparsity_weight * (1.0 + progress * 2.0)
        
        return {
            'sparsity_weight': sparsity_weight,
            'concentration_weight': concentration_weight,
            'entropy_weight': 0.1 * (1.0 + progress),  # Gradually increase diversity
        }
    
    def forward(self, proto_features, pooled_features=None):
        """Adaptive prototype loss"""
        weights = self.get_adaptive_weights()
        
        # Create dynamic prototype loss
        prototype_loss = ImprovedPrototypeLoss(
            weight=1.0,
            sparsity_weight=weights['sparsity_weight'],
            concentration_weight=weights['concentration_weight'],
            entropy_weight=weights['entropy_weight']
        )
        
        return prototype_loss(proto_features, pooled_features)