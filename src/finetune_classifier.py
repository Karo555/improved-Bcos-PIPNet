import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

# Add PIPNet modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PIPNet'))
from pipnet.pipnet import NonNegLinear


class ScoringSheetClassifier(nn.Module):
    """
    Scoring-sheet classification head for fine-tuning B-cos PIP-Net
    """
    def __init__(self, pretrained_bcos_pipnet, num_classes, freeze_prototypes=True):
        super().__init__()
        
        # Copy pretrained components
        self.backbone = pretrained_bcos_pipnet.backbone
        self.prototype_layer = pretrained_bcos_pipnet.prototype_layer
        self.global_pool = pretrained_bcos_pipnet.global_pool
        self.num_prototypes = pretrained_bcos_pipnet.num_prototypes
        self.num_classes = num_classes
        
        # Freeze prototype learning components if specified
        if freeze_prototypes:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.prototype_layer.parameters():
                param.requires_grad = False
        
        # Non-negative linear classification layer (scoring sheet)
        # Weight w_ij indicates relevance of prototype i to class j
        self.classifier = NonNegLinear(self.num_prototypes, num_classes, bias=True)
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier with positive weights (like PIP-Net)"""
        nn.init.normal_(self.classifier.weight, mean=1.0, std=0.1)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.)
    
    def forward(self, x, inference=False):
        """
        Forward pass for classification
        
        Args:
            x: Input tensor of shape (batch_size, 6, H, W)
            inference: Whether in inference mode (applies threshold)
        
        Returns:
            proto_features: Prototype feature maps
            pooled_features: Prototype presence scores (p_i)
            logits: Class logits before softmax
            class_scores: Class confidence scores (after softmax)
        """
        # Extract prototype features
        backbone_features = self.backbone(x)
        proto_features = self.prototype_layer(backbone_features)
        
        # Get prototype presence scores p_i
        pooled_features = self.global_pool(proto_features)
        
        # Apply threshold during inference (like PIP-Net)
        if inference:
            pooled_features = torch.where(pooled_features < 0.1, 0., pooled_features)
        
        # Scoring sheet: class score = sum(p_i * w_ij) for class j
        # where p_i is prototype presence and w_ij is prototype-class weight
        logits = self.classifier(pooled_features)
        
        # Apply softmax to get class confidence scores
        class_scores = F.softmax(logits, dim=1)
        
        return proto_features, pooled_features, logits, class_scores
    
    def get_prototype_class_relevance(self):
        """
        Get the learned relevance weights of prototypes to classes
        
        Returns:
            weight_matrix: (num_prototypes, num_classes) tensor of relevance weights
        """
        return self.classifier.weight.data.t()  # Transpose to get (prototypes, classes)
    
    def get_scoring_sheet_explanation(self, x, class_idx=None):
        """
        Get explanation in terms of prototype contributions to class scores
        
        Args:
            x: Input tensor
            class_idx: Class index to explain (if None, explains predicted class)
        
        Returns:
            explanation: Dictionary with prototype contributions
        """
        self.eval()
        with torch.no_grad():
            proto_features, pooled_features, logits, class_scores = self.forward(x, inference=True)
            
            if class_idx is None:
                class_idx = torch.argmax(class_scores, dim=1)
            
            # Get relevance weights for the specified class
            class_weights = self.classifier.weight[class_idx, :]  # Shape: (batch_size, num_prototypes)
            
            # Calculate contribution of each prototype to the class score
            # contribution_i = p_i * w_ij
            contributions = pooled_features * class_weights
            
            return {
                'prototype_presences': pooled_features,
                'class_weights': class_weights,
                'contributions': contributions,
                'predicted_class': class_idx,
                'class_scores': class_scores,
                'total_class_score': contributions.sum(dim=1)
            }
    
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
        from tqdm import tqdm
        
        self.eval()
        prototype_max_activations = torch.zeros(self.classifier.in_features, device=device)
        
        print(f"Evaluating prototype usage on {len(data_loader.dataset)} samples...")
        
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc="Computing prototype activations"):
                inputs = inputs.to(device)
                
                # Get prototype activations
                _, pooled_features, _, _ = self.forward(inputs, inference=True)
                
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


class FineTuningLoss(nn.Module):
    """
    Combined loss for fine-tuning with additional regularization
    """
    def __init__(self, 
                 nll_weight=1.0,
                 l1_weight=0.0001,
                 orthogonal_weight=0.0):
        super().__init__()
        self.nll_weight = nll_weight
        self.l1_weight = l1_weight
        self.orthogonal_weight = orthogonal_weight
        
        # Negative log-likelihood loss
        self.nll_loss = nn.NLLLoss(reduction='mean')
    
    def forward(self, logits, targets, classifier_weights=None):
        """
        Compute fine-tuning loss
        
        Args:
            logits: Raw class logits
            targets: Ground truth class labels
            classifier_weights: Classifier weight matrix for regularization
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Negative log-likelihood loss
        nll_loss = self.nll_loss(log_probs, targets)
        
        total_loss = self.nll_weight * nll_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'nll_loss': nll_loss
        }
        
        # L1 regularization on classifier weights (encourage sparsity)
        if self.l1_weight > 0 and classifier_weights is not None:
            l1_loss = torch.sum(torch.abs(classifier_weights))
            total_loss += self.l1_weight * l1_loss
            loss_dict['l1_loss'] = l1_loss
        
        # Orthogonal regularization (encourage diverse prototypes)
        if self.orthogonal_weight > 0 and classifier_weights is not None:
            # Encourage orthogonality between prototype weight vectors
            weight_norm = F.normalize(classifier_weights, dim=0)
            correlation_matrix = torch.mm(weight_norm.t(), weight_norm)
            # Penalize off-diagonal correlations
            mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
            orthogonal_loss = torch.sum((correlation_matrix * (1 - mask)) ** 2)
            total_loss += self.orthogonal_weight * orthogonal_loss
            loss_dict['orthogonal_loss'] = orthogonal_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


def evaluate_model(model, test_loader, device, num_classes):
    """
    Evaluate fine-tuned model
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    # For computing prototype usage statistics
    prototype_activations = []
    prototype_usage_count = torch.zeros(model.num_prototypes)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Convert RGB to 6-channel if needed
            if inputs.shape[1] == 3:
                r, g, b = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
                inv_r, inv_g, inv_b = 1.0 - r, 1.0 - g, 1.0 - b
                inputs = torch.cat([r, g, b, inv_r, inv_g, inv_b], dim=1)
            
            proto_features, pooled_features, logits, class_scores = model(inputs, inference=True)
            predicted = torch.argmax(class_scores, dim=1)
            
            # Overall accuracy
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
            
            # Prototype usage statistics
            prototype_activations.append(pooled_features.cpu())
            # Count active prototypes (> 0.1 threshold)
            active_prototypes = (pooled_features > 0.1).sum(dim=0)
            prototype_usage_count += active_prototypes.cpu()
    
    # Calculate metrics
    overall_accuracy = 100. * correct / total
    class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                       for i in range(num_classes)]
    
    # Prototype statistics
    all_prototype_activations = torch.cat(prototype_activations, dim=0)
    
    return {
        'accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'prototype_usage_count': prototype_usage_count,
        'prototype_activations': all_prototype_activations,
        'mean_prototype_activation': all_prototype_activations.mean().item(),
        'active_prototypes_per_sample': (all_prototype_activations > 0.1).sum(dim=1).float().mean().item(),
        'num_unused_prototypes': (prototype_usage_count == 0).sum().item()
    }


def compute_prototype_purity(model, data_loader, device, num_classes):
    """
    Compute prototype purity metrics (simplified version of CUB evaluation)
    
    This measures how consistent each prototype is across samples of the same class
    """
    model.eval()
    
    # Collect prototype activations per class
    class_prototype_activations = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Computing purity'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Convert RGB to 6-channel if needed
            if inputs.shape[1] == 3:
                r, g, b = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
                inv_r, inv_g, inv_b = 1.0 - r, 1.0 - g, 1.0 - b
                inputs = torch.cat([r, g, b, inv_r, inv_g, inv_b], dim=1)
            
            proto_features, pooled_features, _, _ = model(inputs, inference=True)
            
            # Group by class
            for i, target in enumerate(targets):
                class_prototype_activations[target.item()].append(pooled_features[i].cpu())
    
    # Compute purity for each prototype
    prototype_purities = []
    class_prototype_means = {}
    
    for class_idx in range(num_classes):
        if len(class_prototype_activations[class_idx]) > 0:
            class_activations = torch.stack(class_prototype_activations[class_idx])
            class_prototype_means[class_idx] = class_activations.mean(dim=0)
        else:
            class_prototype_means[class_idx] = torch.zeros(model.num_prototypes)
    
    for proto_idx in range(model.num_prototypes):
        # Find which class this prototype is most active for
        class_means_for_proto = [class_prototype_means[c][proto_idx].item() 
                                for c in range(num_classes)]
        best_class = np.argmax(class_means_for_proto)
        max_mean = class_means_for_proto[best_class]
        
        if max_mean > 0:
            # Compute purity as consistency within the best class
            if len(class_prototype_activations[best_class]) > 1:
                class_activations = torch.stack(class_prototype_activations[best_class])
                proto_activations = class_activations[:, proto_idx]
                # Purity = 1 - coefficient of variation
                mean_activation = proto_activations.mean()
                std_activation = proto_activations.std()
                if mean_activation > 0:
                    cv = std_activation / mean_activation
                    purity = max(0, 1 - cv.item())
                else:
                    purity = 0
            else:
                purity = 1.0  # Only one sample, perfect purity
        else:
            purity = 0  # Prototype not active
        
        prototype_purities.append(purity)
    
    return {
        'prototype_purities': prototype_purities,
        'mean_purity': np.mean(prototype_purities),
        'std_purity': np.std(prototype_purities),
        'high_purity_prototypes': sum(1 for p in prototype_purities if p > 0.5),
        'class_prototype_means': class_prototype_means
    }


def create_scoring_sheet_classifier(pretrained_path, num_classes, freeze_prototypes=True):
    """
    Factory function to create scoring sheet classifier from pretrained B-cos PIP-Net
    """
    # Load pretrained model
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Import and create the pretrained model
    from bcos_pipnet import create_bcos_pipnet
    
    pretrained_model = create_bcos_pipnet(
        num_prototypes=checkpoint.get('num_prototypes', 512),
        backbone=checkpoint.get('backbone', 'bcos_resnet18')
    )
    
    # Load pretrained weights
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier
    classifier = ScoringSheetClassifier(
        pretrained_bcos_pipnet=pretrained_model,
        num_classes=num_classes,
        freeze_prototypes=freeze_prototypes
    )
    
    return classifier