import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d


class AlignLoss(nn.Module):
    """
    Alignment loss (La) from B-cos Networks
    Encourages alignment between input and weight vectors in BcosConv layers
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, model):
        """
        Compute alignment loss across all BcosConv2d layers in the model
        """
        align_loss = 0.0
        num_layers = 0
        
        for module in model.modules():
            if isinstance(module, BcosConv2d):
                # Get the weight tensor from the linear layer
                weight = module.linear.weight  # Shape: (out_channels, in_channels*kernel_size^2)
                
                # Compute alignment penalty - encourage weights to be diverse
                # This is a simplified version of the B-cos alignment loss
                weight_flat = weight.view(weight.size(0), -1)
                weight_norm = F.normalize(weight_flat, dim=1)
                
                # Compute cosine similarity matrix between weight vectors
                similarity_matrix = torch.mm(weight_norm, weight_norm.t())
                
                # Remove diagonal (self-similarity)
                mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
                similarity_matrix = similarity_matrix * (1 - mask)
                
                # Penalize high similarity between different filters
                align_loss += torch.sum(similarity_matrix ** 2)
                num_layers += 1
        
        if num_layers > 0:
            align_loss = align_loss / num_layers
        
        return self.weight * align_loss


class TanhLoss(nn.Module):
    """
    Tanh loss (Lt) for prototype learning
    Encourages prototype activations to be either high or low (sparse)
    """
    def __init__(self, weight=1.0, temperature=1.0):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
    
    def forward(self, proto_features):
        """
        Compute tanh loss on prototype features
        
        Args:
            proto_features: Prototype activations of shape (batch_size, num_prototypes, H, W)
        """
        # Apply tanh to encourage binary-like activations
        tanh_activations = torch.tanh(proto_features / self.temperature)
        
        # Compute loss - encourage activations to be close to -1 or 1
        # This pushes activations away from 0 (encourages sparsity)
        tanh_loss = torch.mean((tanh_activations ** 2 - 1) ** 2)
        
        return self.weight * tanh_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning
    Brings positive pairs closer and pushes negative pairs apart
    """
    def __init__(self, temperature=0.07, weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
    
    def forward(self, features1, features2):
        """
        Compute contrastive loss between two views
        
        Args:
            features1, features2: Feature representations of shape (batch_size, feature_dim)
        """
        batch_size = features1.size(0)
        
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)  # (2*batch_size, feature_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Remove self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # Select positive and negative pairs
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # Compute InfoNCE loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return self.weight * loss


class CombinedPretrainingLoss(nn.Module):
    """
    Combined loss for self-supervised pre-training: La + Lt
    """
    def __init__(self, align_weight=1.0, tanh_weight=1.0):
        super().__init__()
        self.align_loss = AlignLoss(weight=align_weight)
        self.tanh_loss = TanhLoss(weight=tanh_weight)
    
    def forward(self, model, proto_features1, proto_features2):
        """
        Compute combined pre-training loss
        
        Args:
            model: The model containing BcosConv2d layers
            proto_features1, proto_features2: Prototype features from two views
        """
        # Alignment loss from model weights
        la = self.align_loss(model)
        
        # Tanh loss from prototype features (average over both views)
        lt1 = self.tanh_loss(proto_features1)
        lt2 = self.tanh_loss(proto_features2)
        lt = (lt1 + lt2) / 2.0
        
        total_loss = la + lt
        
        return {
            'total_loss': total_loss,
            'align_loss': la,
            'tanh_loss': lt
        }


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: Normalized feature representations
        """
        batch_size = z1.size(0)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # 2N x D
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # 2N x 2N
        
        # Create positive mask
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        # Remove self-similarity
        sim_matrix = sim_matrix.masked_fill(torch.eye(2 * batch_size, device=z.device).bool(), float('-inf'))
        
        # Compute InfoNCE loss
        pos_sim = sim_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = sim_matrix[~pos_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss