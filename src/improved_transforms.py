import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


class ImprovedSixChannelTransform(nn.Module):
    """
    Improved 6-channel transformation with better normalization
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, img):
        """
        Args:
            img: RGB tensor of shape (C, H, W) with values in [0, 1]
        Returns:
            6-channel tensor of shape (6, H, W)
        """
        if img.dim() == 3 and img.shape[0] == 3:
            # Extract RGB channels
            r, g, b = img[0:1], img[1:2], img[2:3]
            
            # Create improved complementary channels with better normalization
            inv_r = 1.0 - r
            inv_g = 1.0 - g
            inv_b = 1.0 - b
            
            # Optional: Add slight noise to prevent exact complementarity
            # This can help with learning distinct patterns
            noise_scale = 0.01
            inv_r = inv_r + torch.randn_like(inv_r) * noise_scale
            inv_g = inv_g + torch.randn_like(inv_g) * noise_scale
            inv_b = inv_b + torch.randn_like(inv_b) * noise_scale
            
            # Clamp to valid range
            inv_r = torch.clamp(inv_r, 0, 1)
            inv_g = torch.clamp(inv_g, 0, 1)
            inv_b = torch.clamp(inv_b, 0, 1)
            
            # Concatenate to create 6-channel representation
            six_channel = torch.cat([r, g, b, inv_r, inv_g, inv_b], dim=0)
            return six_channel
        else:
            raise ValueError(f"Expected 3-channel RGB image, got shape {img.shape}")


class PrototypeAugmentation:
    """
    Augmentation strategy specifically designed for prototype learning
    """
    def __init__(self, img_size=224, strong_aug_prob=0.8):
        self.img_size = img_size
        self.strong_aug_prob = strong_aug_prob
        
        # Geometric augmentations
        self.geometric_augs = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
        
        # Color augmentations
        self.color_augs = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Advanced augmentations
        self.advanced_augs = [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([self._random_cutout], p=0.3),
            transforms.RandomApply([self._random_noise], p=0.2),
        ]
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        self.six_channel = ImprovedSixChannelTransform()
    
    def _random_cutout(self, img):
        """Random cutout augmentation"""
        if isinstance(img, torch.Tensor):
            img_pil = TF.to_pil_image(img)
        else:
            img_pil = img
            
        w, h = img_pil.size
        cutout_size = min(w, h) // 8
        
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        
        img_array = np.array(img_pil)
        img_array[y:y+cutout_size, x:x+cutout_size] = 128  # Gray cutout
        
        return Image.fromarray(img_array)
    
    def _random_noise(self, img):
        """Add random noise"""
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * 0.05
            return torch.clamp(img + noise, 0, 1)
        else:
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.05, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            return Image.fromarray((img_array * 255).astype(np.uint8))
    
    def __call__(self, img):
        """
        Apply prototype-specific augmentations
        """
        # Convert to PIL if tensor
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Apply geometric augmentations
        img_aug = self.geometric_augs(img)
        
        # Apply color augmentations
        img_aug = self.color_augs(img_aug)
        
        # Apply advanced augmentations with probability
        if random.random() < self.strong_aug_prob:
            for aug in self.advanced_augs:
                img_aug = aug(img_aug)
        
        # Convert to tensor and normalize
        img_tensor = TF.to_tensor(img_aug)
        img_normalized = self.normalize(img_tensor)
        
        # Convert to 6-channel
        img_6ch = self.six_channel(img_normalized)
        
        return img_6ch


class DualViewAugmentation:
    """
    Create two different augmented views for contrastive learning
    Optimized for prototype learning
    """
    def __init__(self, img_size=224, view_difference=0.3):
        self.img_size = img_size
        self.view_difference = view_difference
        
        # View 1: Moderate augmentation
        self.view1_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
            ], p=0.7),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 1.5))
            ], p=0.3),
            transforms.ToTensor(),
        ])
        
        # View 2: Different augmentation strategy
        self.view2_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.15)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 1.5))
            ], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
        ])
        
        # Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # 6-channel transform
        self.six_channel = ImprovedSixChannelTransform()
    
    def __call__(self, img):
        """
        Returns two different augmented views
        """
        # Apply different transforms
        view1 = self.view1_transform(img)
        view2 = self.view2_transform(img)
        
        # Normalize
        view1 = self.normalize(view1)
        view2 = self.normalize(view2)
        
        # Convert to 6-channel
        view1_6ch = self.six_channel(view1)
        view2_6ch = self.six_channel(view2)
        
        return view1_6ch, view2_6ch


class AdaptiveAugmentation:
    """
    Adaptive augmentation that changes based on training progress
    """
    def __init__(self, img_size=224, total_epochs=200):
        self.img_size = img_size
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Base transforms
        self.base_geometric = transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0))
        self.base_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.six_channel = ImprovedSixChannelTransform()
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_augmentation_strength(self):
        """Get augmentation strength based on training progress"""
        progress = self.current_epoch / self.total_epochs
        
        if progress < 0.1:
            # Early training: mild augmentation
            return {
                'color_jitter_strength': 0.2,
                'blur_prob': 0.1,
                'grayscale_prob': 0.1,
                'cutout_prob': 0.0,
                'noise_prob': 0.0
            }
        elif progress < 0.5:
            # Mid training: moderate augmentation
            return {
                'color_jitter_strength': 0.3,
                'blur_prob': 0.3,
                'grayscale_prob': 0.2,
                'cutout_prob': 0.1,
                'noise_prob': 0.1
            }
        else:
            # Late training: strong augmentation
            return {
                'color_jitter_strength': 0.4,
                'blur_prob': 0.5,
                'grayscale_prob': 0.3,
                'cutout_prob': 0.2,
                'noise_prob': 0.2
            }
    
    def __call__(self, img):
        """Apply adaptive augmentation"""
        strength = self.get_augmentation_strength()
        
        # Geometric augmentations
        img = self.base_geometric(img)
        img = self.base_flip(img)
        
        # Color augmentations
        if random.random() < 0.8:
            img = transforms.ColorJitter(
                brightness=strength['color_jitter_strength'],
                contrast=strength['color_jitter_strength'],
                saturation=strength['color_jitter_strength'],
                hue=strength['color_jitter_strength'] * 0.25
            )(img)
        
        # Blur
        if random.random() < strength['blur_prob']:
            img = transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))(img)
        
        # Grayscale
        if random.random() < strength['grayscale_prob']:
            img = transforms.Grayscale(num_output_channels=3)(img)
        
        # Convert to tensor
        img = self.to_tensor(img)
        
        # Cutout
        if random.random() < strength['cutout_prob']:
            img = self._apply_cutout(img)
        
        # Noise
        if random.random() < strength['noise_prob']:
            img = self._apply_noise(img)
        
        # Normalize and convert to 6-channel
        img = self.normalize(img)
        img_6ch = self.six_channel(img)
        
        return img_6ch
    
    def _apply_cutout(self, img):
        """Apply cutout augmentation"""
        _, h, w = img.shape
        cutout_size = min(h, w) // 8
        
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        
        img[:, y:y+cutout_size, x:x+cutout_size] = 0.5  # Gray cutout
        return img
    
    def _apply_noise(self, img):
        """Apply noise augmentation"""
        noise = torch.randn_like(img) * 0.05
        return torch.clamp(img + noise, 0, 1)


class CIFARPrototypeAugmentation:
    """
    CIFAR-10 specific prototype augmentation
    """
    def __init__(self, img_size=32):
        self.img_size = img_size
        
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),  # Less aggressive crop for CIFAR
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
            ], p=0.7),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))  # Smaller kernel for CIFAR
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.15)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([self._random_cutout], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        self.six_channel = ImprovedSixChannelTransform()
    
    def _random_cutout(self, img):
        """CIFAR-specific cutout"""
        if isinstance(img, torch.Tensor):
            img_pil = TF.to_pil_image(img)
        else:
            img_pil = img
            
        w, h = img_pil.size
        cutout_size = 8  # Fixed size for CIFAR
        
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        
        img_array = np.array(img_pil)
        img_array[y:y+cutout_size, x:x+cutout_size] = [128, 128, 128]  # Gray cutout
        
        return Image.fromarray(img_array)
    
    def __call__(self, img):
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        
        view1_6ch = self.six_channel(view1)
        view2_6ch = self.six_channel(view2)
        
        return view1_6ch, view2_6ch


def get_improved_augmentation(dataset='cifar10', img_size=None, augmentation_type='adaptive'):
    """
    Factory function for improved augmentations
    """
    if dataset == 'cifar10':
        img_size = img_size or 32
        if augmentation_type == 'prototype':
            return PrototypeAugmentation(img_size)
        elif augmentation_type == 'dual_view':
            return CIFARPrototypeAugmentation(img_size)
        else:  # adaptive
            return AdaptiveAugmentation(img_size)
    
    elif dataset == 'cub':
        img_size = img_size or 224
        if augmentation_type == 'prototype':
            return PrototypeAugmentation(img_size)
        elif augmentation_type == 'dual_view':
            return DualViewAugmentation(img_size)
        else:  # adaptive
            return AdaptiveAugmentation(img_size)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


class MixUpAugmentation:
    """
    MixUp augmentation for prototype learning
    """
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y=None):
        """
        Apply MixUp to a batch
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        
        if batch_y is not None:
            y_a, y_b = batch_y, batch_y[index]
            return mixed_x, y_a, y_b, lam
        else:
            return mixed_x, lam