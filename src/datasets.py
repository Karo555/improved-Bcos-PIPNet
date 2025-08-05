import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
import os
from PIL import Image
from transforms import ContrastiveAugmentation, ContrastiveAugmentationCIFAR


class ContrastiveDataset(Dataset):
    """
    Dataset wrapper for contrastive learning
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.base_dataset, 'data'):
            # For CIFAR-10
            img = Image.fromarray(self.base_dataset.data[idx])
            if hasattr(self.base_dataset, 'targets'):
                label = self.base_dataset.targets[idx]
            else:
                label = self.base_dataset.labels[idx]
        else:
            # For ImageFolder datasets
            img, label = self.base_dataset[idx]
        
        # Generate two augmented views
        view1, view2 = self.transform(img)
        
        return view1, view2, label


class CIFAR10ContrastiveDataset:
    """
    CIFAR-10 dataset setup for contrastive learning
    """
    def __init__(self, data_dir='./data', batch_size=128, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Contrastive augmentation for CIFAR-10
        self.transform = ContrastiveAugmentationCIFAR(img_size=32)
        
        # Base transform for evaluation
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
    
    def get_dataloaders(self):
        # Download CIFAR-10
        train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None  # We'll apply transforms in ContrastiveDataset
        )
        
        test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.eval_transform
        )
        
        # Wrap with contrastive dataset
        contrastive_train_dataset = ContrastiveDataset(train_dataset, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


class CUBContrastiveDataset:
    """
    CUB-200-2011 dataset setup for contrastive learning
    """
    def __init__(self, data_dir, batch_size=64, num_workers=4, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
        # Contrastive augmentation for CUB
        self.transform = ContrastiveAugmentation(img_size=img_size)
        
        # Base transform for evaluation
        self.eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self):
        # CUB dataset should be organized as ImageFolder
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise ValueError(f"CUB dataset not found in {self.data_dir}. "
                           "Please organize as train/test folders or use preprocess_cub.py")
        
        train_dataset = ImageFolder(train_dir, transform=None)
        test_dataset = ImageFolder(test_dir, transform=self.eval_transform)
        
        # Wrap with contrastive dataset
        contrastive_train_dataset = ContrastiveDataset(train_dataset, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


def get_dataset(dataset_name, **kwargs):
    """
    Factory function to get dataset
    """
    if dataset_name.lower() == 'cifar10':
        return CIFAR10ContrastiveDataset(**kwargs)
    elif dataset_name.lower() == 'cub':
        return CUBContrastiveDataset(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class SixChannelDataset(Dataset):
    """
    Dataset that converts RGB images to 6-channel format
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        
        # Convert to 6-channel if it's RGB
        if img.shape[0] == 3:
            r, g, b = img[0:1], img[1:2], img[2:3]
            inv_r, inv_g, inv_b = 1.0 - r, 1.0 - g, 1.0 - b
            img = torch.cat([r, g, b, inv_r, inv_g, inv_b], dim=0)
        
        return img, label