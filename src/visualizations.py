import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys

# Add B-cos modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'B-cos'))
from modules.bcosconv2d import BcosConv2d


class BcosPIPNetVisualizer:
    """
    Comprehensive visualization toolkit for B-cos PIP-Net
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Store intermediate activations
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # Register hooks for B-cos explanations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for B-cos layers"""
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for B-cos layers
        for name, module in self.model.named_modules():
            if isinstance(module, BcosConv2d):
                handle_forward = module.register_forward_hook(save_activation(name))
                handle_backward = module.register_backward_hook(save_gradient(name))
                self.hooks.extend([handle_forward, handle_backward])
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def denormalize_image(self, tensor, dataset='cifar10'):
        """Denormalize image tensor for visualization"""
        if dataset == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        elif dataset == 'cub':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Move to same device as tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        
        # Denormalize
        denorm = tensor * std + mean
        return torch.clamp(denorm, 0, 1)
    
    def extract_rgb_from_6channel(self, tensor_6ch):
        """Extract RGB channels from 6-channel input"""
        return tensor_6ch[:, :3, :, :]  # Take first 3 channels (r, g, b)
    
    def visualize_input_image(self, image_6ch, dataset='cifar10', title="Input Image"):
        """
        Visualize input image from 6-channel tensor
        
        Args:
            image_6ch: 6-channel input tensor (1, 6, H, W)
            dataset: Dataset name for denormalization
            title: Plot title
        """
        # Extract RGB channels
        rgb_tensor = self.extract_rgb_from_6channel(image_6ch)
        
        # Denormalize
        rgb_denorm = self.denormalize_image(rgb_tensor, dataset)
        
        # Convert to numpy and transpose
        img_np = rgb_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        return img_np
    
    def get_prototype_activations(self, image_6ch):
        """
        Get prototype activations and locations for an image
        
        Returns:
            proto_features: Prototype feature maps (1, num_prototypes, H, W)
            pooled_features: Global pooled features (1, num_prototypes)
            activation_locations: Peak activation locations for each prototype
        """
        with torch.no_grad():
            proto_features, pooled_features, _, _ = self.model(image_6ch.to(self.device), inference=True)
            
        # Find peak activation locations
        activation_locations = []
        for i in range(proto_features.shape[1]):  # For each prototype
            proto_map = proto_features[0, i]  # (H, W)
            flat_idx = torch.argmax(proto_map.flatten())
            h, w = flat_idx // proto_map.shape[1], flat_idx % proto_map.shape[1]
            max_val = proto_map[h, w].item()
            activation_locations.append((h.item(), w.item(), max_val))
        
        return proto_features, pooled_features, activation_locations
    
    def visualize_prototype_activations(self, image_6ch, top_k=9, dataset='cifar10', 
                                      threshold=0.1, figsize=(15, 10)):
        """
        Visualize top-k prototype activations on the input image
        
        Args:
            image_6ch: 6-channel input tensor
            top_k: Number of top prototypes to show
            dataset: Dataset name for denormalization
            threshold: Minimum activation threshold
            figsize: Figure size
        """
        # Get prototype activations
        proto_features, pooled_features, locations = self.get_prototype_activations(image_6ch)
        
        # Get top-k active prototypes
        pooled_np = pooled_features.squeeze(0).cpu().numpy()
        active_indices = np.where(pooled_np > threshold)[0]
        
        if len(active_indices) == 0:
            print("No prototypes above threshold!")
            return
        
        # Sort by activation strength
        sorted_indices = active_indices[np.argsort(pooled_np[active_indices])[::-1]]
        top_prototypes = sorted_indices[:top_k]
        
        # Prepare base image
        rgb_tensor = self.extract_rgb_from_6channel(image_6ch)
        img_denorm = self.denormalize_image(rgb_tensor, dataset)
        base_img = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Create visualization
        n_cols = 3
        n_rows = int(np.ceil(len(top_prototypes) / n_cols))
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        for idx, proto_idx in enumerate(top_prototypes):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Show base image
            ax.imshow(base_img)
            
            # Overlay prototype activation heatmap
            proto_map = proto_features[0, proto_idx].cpu().numpy()
            proto_map_resized = cv2.resize(proto_map, (base_img.shape[1], base_img.shape[0]))
            
            # Create heatmap overlay
            heatmap = plt.cm.jet(proto_map_resized / (proto_map_resized.max() + 1e-8))
            heatmap[..., 3] = 0.6 * (proto_map_resized / (proto_map_resized.max() + 1e-8))  # Alpha channel
            
            ax.imshow(heatmap, alpha=0.6)
            
            # Mark peak location
            h, w, max_val = locations[proto_idx]
            scale_h = base_img.shape[0] / proto_features.shape[2]
            scale_w = base_img.shape[1] / proto_features.shape[3]
            peak_y, peak_x = h * scale_h, w * scale_w
            
            ax.plot(peak_x, peak_y, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
            
            ax.set_title(f'P{proto_idx}: {pooled_np[proto_idx]:.3f}', fontsize=10)
            ax.axis('off')
        
        plt.suptitle('Top Prototype Activations', fontsize=16)
        plt.tight_layout()
        return top_prototypes, pooled_np[top_prototypes]
    
    def get_bcos_contributions(self, image_6ch, target_class=None):
        """
        Get B-cos spatial contribution maps using gradients
        
        Args:
            image_6ch: Input 6-channel tensor
            target_class: Target class for explanation (if None, use predicted class)
        
        Returns:
            contribution_maps: Dictionary of contribution maps from B-cos layers
            predicted_class: Predicted class index
            class_score: Confidence score for predicted class
        """
        # Clear previous activations
        self.activations.clear()
        self.gradients.clear()
        
        # Enable gradients
        image_6ch.requires_grad_(True)
        
        # Forward pass
        proto_features, pooled_features, logits, class_scores = self.model(image_6ch.to(self.device))
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(class_scores, dim=1)
        else:
            target_class = torch.tensor([target_class]).to(self.device)
        
        predicted_class = target_class.item()
        class_score = class_scores[0, predicted_class].item()
        
        # Backward pass
        class_output = class_scores[0, target_class]
        class_output.backward(retain_graph=True)
        
        # Generate contribution maps
        contribution_maps = {}
        
        for name, activation in self.activations.items():
            if name in self.gradients:
                gradient = self.gradients[name]
                
                # Compute contribution as activation * gradient
                contribution = activation * gradient
                
                # Sum over channels to get spatial contribution
                spatial_contribution = contribution.sum(dim=1, keepdim=True)  # (1, 1, H, W)
                
                contribution_maps[name] = spatial_contribution.squeeze(0).squeeze(0).cpu().numpy()
        
        return contribution_maps, predicted_class, class_score
    
    def visualize_bcos_contributions(self, image_6ch, target_class=None, dataset='cifar10',
                                   layer_names=None, figsize=(20, 15)):
        """
        Visualize B-cos spatial contribution maps
        
        Args:
            image_6ch: Input 6-channel tensor
            target_class: Target class for explanation
            dataset: Dataset name for denormalization
            layer_names: Specific layers to visualize (if None, show all)
            figsize: Figure size
        """
        # Get contributions
        contributions, pred_class, class_score = self.get_bcos_contributions(image_6ch, target_class)
        
        if len(contributions) == 0:
            print("No B-cos contributions found!")
            return
        
        # Filter layers if specified
        if layer_names is not None:
            contributions = {k: v for k, v in contributions.items() if k in layer_names}
        
        # Prepare base image
        rgb_tensor = self.extract_rgb_from_6channel(image_6ch)
        img_denorm = self.denormalize_image(rgb_tensor, dataset)
        base_img = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Create visualization
        n_layers = len(contributions)
        n_cols = min(4, n_layers + 1)  # +1 for original image
        n_rows = int(np.ceil((n_layers + 1) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Show original image
        axes[0, 0].imshow(base_img)
        axes[0, 0].set_title(f'Input\nPred: Class {pred_class}\nConf: {class_score:.3f}')
        axes[0, 0].axis('off')
        
        # Show contribution maps
        for idx, (layer_name, contrib_map) in enumerate(contributions.items()):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols
            
            # Resize contribution map to match image size
            contrib_resized = cv2.resize(contrib_map, (base_img.shape[1], base_img.shape[0]))
            
            # Show base image
            axes[row, col].imshow(base_img, alpha=0.7)
            
            # Overlay contribution map
            # Use RdBu colormap: red for positive, blue for negative
            vmax = np.abs(contrib_resized).max()
            vmin = -vmax
            
            im = axes[row, col].imshow(contrib_resized, cmap='RdBu_r', alpha=0.8, 
                                     vmin=vmin, vmax=vmax)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Clean up layer name for display
            display_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            axes[row, col].set_title(f'{display_name}\nContribution Map', fontsize=10)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(n_layers + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('B-cos Spatial Contribution Maps', fontsize=16)
        plt.tight_layout()
        
        return contributions, pred_class, class_score
    
    def create_comprehensive_visualization(self, image_6ch, target_class=None, 
                                         dataset='cifar10', class_names=None,
                                         top_k_prototypes=6, figsize=(20, 15)):
        """
        Create comprehensive visualization showing:
        1) Input image
        2) Top prototype activations
        3) B-cos contribution maps
        
        Args:
            image_6ch: Input 6-channel tensor
            target_class: Target class for explanation
            dataset: Dataset name
            class_names: List of class names
            top_k_prototypes: Number of top prototypes to show
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1) Input image
        ax_input = fig.add_subplot(gs[0, 0])
        rgb_tensor = self.extract_rgb_from_6channel(image_6ch)
        img_denorm = self.denormalize_image(rgb_tensor, dataset)
        base_img = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        ax_input.imshow(base_img)
        ax_input.set_title('Input Image', fontsize=12, fontweight='bold')
        ax_input.axis('off')
        
        # Get model predictions
        with torch.no_grad():
            proto_features, pooled_features, logits, class_scores = self.model(image_6ch.to(self.device))
        
        pred_class = torch.argmax(class_scores, dim=1).item()
        class_score = class_scores[0, pred_class].item()
        
        if class_names and pred_class < len(class_names):
            pred_name = class_names[pred_class]
        else:
            pred_name = f"Class {pred_class}"
        
        # Add prediction info
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_pred.text(0.1, 0.7, f'Prediction:', fontsize=12, fontweight='bold')
        ax_pred.text(0.1, 0.5, pred_name, fontsize=14, color='red')
        ax_pred.text(0.1, 0.3, f'Confidence: {class_score:.3f}', fontsize=12)
        ax_pred.axis('off')
        
        # 2) Top prototype activations
        proto_features, pooled_features, locations = self.get_prototype_activations(image_6ch)
        pooled_np = pooled_features.squeeze(0).cpu().numpy()
        
        # Get top prototypes
        active_indices = np.where(pooled_np > 0.1)[0]
        if len(active_indices) > 0:
            sorted_indices = active_indices[np.argsort(pooled_np[active_indices])[::-1]]
            top_prototypes = sorted_indices[:top_k_prototypes]
            
            # Show top prototypes
            for i, proto_idx in enumerate(top_prototypes):
                if i >= 6:  # Limit to 6 prototypes for space
                    break
                    
                row = 1 + i // 3
                col = i % 3
                
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(base_img)
                
                # Overlay prototype heatmap
                proto_map = proto_features[0, proto_idx].cpu().numpy()
                proto_map_resized = cv2.resize(proto_map, (base_img.shape[1], base_img.shape[0]))
                
                heatmap = plt.cm.jet(proto_map_resized / (proto_map_resized.max() + 1e-8))
                heatmap[..., 3] = 0.6 * (proto_map_resized / (proto_map_resized.max() + 1e-8))
                
                ax.imshow(heatmap, alpha=0.6)
                ax.set_title(f'P{proto_idx}: {pooled_np[proto_idx]:.3f}', fontsize=10)
                ax.axis('off')
        
        # 3) B-cos contributions
        contributions, _, _ = self.get_bcos_contributions(image_6ch, target_class)
        
        if contributions:
            # Show one main contribution map (from the last B-cos layer)
            layer_names = list(contributions.keys())
            main_layer = layer_names[-1] if layer_names else None
            
            if main_layer:
                ax_contrib = fig.add_subplot(gs[0, 2:])
                
                contrib_map = contributions[main_layer]
                contrib_resized = cv2.resize(contrib_map, (base_img.shape[1], base_img.shape[0]))
                
                ax_contrib.imshow(base_img, alpha=0.7)
                
                vmax = np.abs(contrib_resized).max()
                vmin = -vmax
                
                im = ax_contrib.imshow(contrib_resized, cmap='RdBu_r', alpha=0.8, 
                                     vmin=vmin, vmax=vmax)
                
                plt.colorbar(im, ax=ax_contrib, fraction=0.046, pad=0.04)
                ax_contrib.set_title('B-cos Spatial Contributions\n(Red: Positive, Blue: Negative)', 
                                   fontsize=12, fontweight='bold')
                ax_contrib.axis('off')
        
        plt.suptitle('B-cos PIP-Net Comprehensive Visualization', fontsize=16, fontweight='bold')
        
        return {
            'prediction': pred_name,
            'confidence': class_score,
            'top_prototypes': top_prototypes if 'top_prototypes' in locals() else [],
            'prototype_scores': pooled_np[top_prototypes] if 'top_prototypes' in locals() else [],
            'contributions': contributions
        }


def create_prototype_comparison_grid(visualizer, images_6ch, titles=None, 
                                   dataset='cifar10', figsize=(20, 15)):
    """
    Create a grid comparing prototype activations across multiple images
    
    Args:
        visualizer: BcosPIPNetVisualizer instance
        images_6ch: List of 6-channel image tensors
        titles: List of titles for each image
        dataset: Dataset name
        figsize: Figure size
    """
    n_images = len(images_6ch)
    
    fig, axes = plt.subplots(n_images, 4, figsize=figsize)
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, image_6ch in enumerate(images_6ch):
        # Show original image
        rgb_tensor = visualizer.extract_rgb_from_6channel(image_6ch)
        img_denorm = visualizer.denormalize_image(rgb_tensor, dataset)
        base_img = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        axes[i, 0].imshow(base_img)
        title = titles[i] if titles and i < len(titles) else f'Image {i+1}'
        axes[i, 0].set_title(title)
        axes[i, 0].axis('off')
        
        # Get top 3 prototypes
        proto_features, pooled_features, _ = visualizer.get_prototype_activations(image_6ch)
        pooled_np = pooled_features.squeeze(0).cpu().numpy()
        
        active_indices = np.where(pooled_np > 0.1)[0]
        if len(active_indices) > 0:
            sorted_indices = active_indices[np.argsort(pooled_np[active_indices])[::-1]]
            top_3 = sorted_indices[:3]
            
            for j, proto_idx in enumerate(top_3):
                axes[i, j+1].imshow(base_img)
                
                proto_map = proto_features[0, proto_idx].cpu().numpy()
                proto_map_resized = cv2.resize(proto_map, (base_img.shape[1], base_img.shape[0]))
                
                heatmap = plt.cm.jet(proto_map_resized / (proto_map_resized.max() + 1e-8))
                heatmap[..., 3] = 0.6 * (proto_map_resized / (proto_map_resized.max() + 1e-8))
                
                axes[i, j+1].imshow(heatmap, alpha=0.6)
                axes[i, j+1].set_title(f'P{proto_idx}: {pooled_np[proto_idx]:.3f}')
                axes[i, j+1].axis('off')
        else:
            for j in range(1, 4):
                axes[i, j].text(0.5, 0.5, 'No active\nprototypes', 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.suptitle('Prototype Activation Comparison', fontsize=16)
    plt.tight_layout()
    
    return fig