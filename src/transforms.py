import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random


class SixChannelTransform(nn.Module):
    """
    Transform RGB image to 6-channel representation: [r,g,b,1-r,1-g,1-b]
    """
    def __init__(self):
        super().__init__()
        
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
            
            # Create complementary channels
            inv_r = 1.0 - r
            inv_g = 1.0 - g
            inv_b = 1.0 - b
            
            # Concatenate to create 6-channel representation
            six_channel = torch.cat([r, g, b, inv_r, inv_g, inv_b], dim=0)
            return six_channel
        else:
            raise ValueError(f"Expected 3-channel RGB image, got shape {img.shape}")


class ContrastiveAugmentation:
    """
    Creates positive pairs through different augmentations
    """
    def __init__(self, img_size=224):
        self.img_size = img_size
        
        # Strong augmentation for first view
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Different augmentation for second view
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 6-channel transform
        self.six_channel = SixChannelTransform()
    
    def __call__(self, img):
        """
        Returns two different augmented views of the same image in 6-channel format
        """
        # Apply different augmentations to create positive pair
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        
        # Convert to 6-channel representation
        view1_6ch = self.six_channel(view1)
        view2_6ch = self.six_channel(view2)
        
        return view1_6ch, view2_6ch


class ContrastiveAugmentationCIFAR:
    """
    CIFAR-10 specific contrastive augmentation
    """
    def __init__(self, img_size=32):
        self.img_size = img_size
        
        # CIFAR-10 specific augmentations
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        self.six_channel = SixChannelTransform()
    
    def __call__(self, img):
        view1 = self.transform1(img)
        view2 = self.transform2(img)
        
        view1_6ch = self.six_channel(view1)
        view2_6ch = self.six_channel(view2)
        
        return view1_6ch, view2_6ch