import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm
import json
import numpy as np

from improved_bcos_pipnet import (
    create_improved_bcos_pipnet, 
    ProgressiveTrainingStrategy,
    PrototypeMonitor,
    debug_prototype_learning
)
from improved_losses import ImprovedCombinedPretrainingLoss, AdaptivePrototypeLoss
from improved_transforms import get_improved_augmentation
from datasets import get_dataset
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Improved self-supervised pre-training of B-cos PIP-Net')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cub'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='bcos_resnet18',
                        choices=['bcos_resnet18', 'bcos_resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--num_prototypes', type=int, default=256,  # Reduced for better learning
                        help='Number of prototypes')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Warmup epochs for progressive training')
    parser.add_argument('--freeze_epochs', type=int, default=60,
                        help='Epochs to keep backbone frozen')
    
    # Learning rates
    parser.add_argument('--lr_prototype', type=float, default=3e-3,  # Higher for prototype learning
                        help='Learning rate for prototype layers')
    parser.add_argument('--lr_backbone', type=float, default=1e-4,   # Lower for backbone
                        help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Loss weights (improved defaults)
    parser.add_argument('--align_weight', type=float, default=0.5,    # Reduced
                        help='Weight for alignment loss')
    parser.add_argument('--prototype_weight', type=float, default=3.0, # Increased
                        help='Weight for prototype loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='Weight for contrastive loss')
    parser.add_argument('--consistency_weight', type=float, default=1.0, # New
                        help='Weight for consistency loss')
    
    # Training strategy
    parser.add_argument('--progressive_training', action='store_true', default=True,
                        help='Use progressive training strategy')
    parser.add_argument('--adaptive_augmentation', action='store_true', default=True,
                        help='Use adaptive augmentation')
    parser.add_argument('--monitor_prototypes', action='store_true', default=True,
                        help='Monitor prototype learning progress')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./logs/improved_pretraining',
                        help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/improved_pretraining',
                        help='Save directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log interval')
    parser.add_argument('--debug_interval', type=int, default=20,
                        help='Prototype debugging interval')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_improved_dataloaders(args):
    """Create improved dataloaders with better augmentation"""
    if args.dataset == 'cifar10':
        # Use improved CIFAR augmentation
        from improved_transforms import CIFARPrototypeAugmentation
        from datasets import ContrastiveDataset
        import torchvision
        
        # Base dataset
        train_dataset_base = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=None
        )
        
        # Improved augmentation
        augmentation = CIFARPrototypeAugmentation(img_size=32)
        
        # Contrastive dataset
        train_dataset = ContrastiveDataset(train_dataset_base, augmentation)
        
        # Test dataset (for monitoring)
        test_transform = get_improved_augmentation('cifar10', augmentation_type='prototype')
        test_dataset_base = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=test_transform
        )
        
        from datasets import SixChannelDataset
        test_dataset = SixChannelDataset(test_dataset_base)
        
    else:  # CUB
        dataset_handler = get_dataset('cub',
                                    data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    img_size=224)
        return dataset_handler.get_dataloaders()
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def create_improved_optimizer(model, args, training_config):
    """Create optimizer with different learning rates for different components"""
    prototype_params = []
    backbone_params = []
    
    # Separate parameters
    for name, param in model.named_parameters():
        if 'prototype_layer' in name or 'projection_head' in name:
            prototype_params.append(param)
        else:
            backbone_params.append(param)
    
    # Learning rate multiplier from progressive training
    lr_mult = training_config.get('learning_rate_multiplier', 1.0)
    
    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {
            'params': prototype_params, 
            'lr': args.lr_prototype * lr_mult,
            'weight_decay': args.weight_decay
        },
        {
            'params': backbone_params, 
            'lr': args.lr_backbone * lr_mult,
            'weight_decay': args.weight_decay * 0.1  # Less regularization for backbone
        }
    ])
    
    return optimizer


def train_epoch_improved(model, train_loader, criterion, optimizer, epoch, args, 
                        writer, training_strategy, prototype_monitor):
    model.train()
    
    # Get training configuration
    training_config = training_strategy.get_training_config()
    
    # Freeze/unfreeze backbone based on strategy
    if training_config['freeze_backbone']:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()
    
    # Update loss weights based on training phase
    criterion_weights = {
        'align_weight': args.align_weight * training_config['align_weight'],
        'prototype_weight': args.prototype_weight * training_config['prototype_weight'],
        'contrastive_weight': args.contrastive_weight * training_config['contrastive_weight'],
        'consistency_weight': args.consistency_weight
    }
    
    # Update criterion with new weights
    criterion.align_loss.weight = criterion_weights['align_weight']
    criterion.prototype_loss.weight = criterion_weights['prototype_weight']
    criterion.contrastive_loss.weight = criterion_weights['contrastive_weight']
    criterion.consistency_weight = criterion_weights['consistency_weight']
    
    total_loss = 0.0
    total_align_loss = 0.0
    total_prototype_loss = 0.0
    total_contrastive_loss = 0.0
    total_consistency_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} ({training_config["phase"]})')
    
    for batch_idx, (view1, view2, labels) in enumerate(pbar):
        view1, view2 = view1.to(args.device), view2.to(args.device)
        
        optimizer.zero_grad()
        
        # Forward pass through both views
        proto_features1, pooled1, projected1 = model(view1, return_features=True)
        proto_features2, pooled2, projected2 = model(view2, return_features=True)
        
        # Compute improved combined loss
        loss_dict = criterion(model, proto_features1, proto_features2, pooled1, pooled2)
        loss = loss_dict['total_loss']
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
            print(f"Loss components: {loss_dict}")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_align_loss += loss_dict['align_loss'].item()
        total_prototype_loss += loss_dict['prototype_loss'].item()
        total_contrastive_loss += loss_dict['contrastive_loss'].item()
        total_consistency_loss += loss_dict['consistency_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'La': f'{loss_dict["align_loss"].item():.4f}',
            'Lp': f'{loss_dict["prototype_loss"].item():.4f}',
            'Lc': f'{loss_dict["contrastive_loss"].item():.4f}',
            'Phase': training_config['phase'][:8]
        })
        
        # Log to tensorboard
        if writer and batch_idx % args.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Align', loss_dict['align_loss'].item(), global_step)
            writer.add_scalar('Loss/Prototype', loss_dict['prototype_loss'].item(), global_step)
            writer.add_scalar('Loss/Contrastive', loss_dict['contrastive_loss'].item(), global_step)
            writer.add_scalar('Loss/Consistency', loss_dict['consistency_loss'].item(), global_step)
            
            # Log prototype details
            proto_details = loss_dict['prototype_details']
            writer.add_scalar('Prototype/Sparsity_Loss', proto_details['sparsity_loss'].item(), global_step)
            writer.add_scalar('Prototype/Concentration_Loss', proto_details['concentration_loss'].item(), global_step)
            writer.add_scalar('Prototype/Entropy_Loss', proto_details['entropy_loss'].item(), global_step)
            writer.add_scalar('Prototype/Diversity_Loss', proto_details['diversity_loss'].item(), global_step)
    
    return {
        'total_loss': total_loss / len(train_loader),
        'align_loss': total_align_loss / len(train_loader),
        'prototype_loss': total_prototype_loss / len(train_loader),
        'contrastive_loss': total_contrastive_loss / len(train_loader),
        'consistency_loss': total_consistency_loss / len(train_loader),
        'training_phase': training_config['phase']
    }


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        args.device = 'cpu'
        print("Using CPU")
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.save_dir, 'improved_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create improved dataloaders
    print(f"Setting up improved {args.dataset} dataset...")
    train_loader, test_loader = get_improved_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create improved model
    print(f"Creating improved {args.backbone} model with {args.num_prototypes} prototypes...")
    model = create_improved_bcos_pipnet(
        num_prototypes=args.num_prototypes,
        backbone=args.backbone,
        pretrained=False,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create improved loss function
    criterion = ImprovedCombinedPretrainingLoss(
        align_weight=args.align_weight,
        prototype_weight=args.prototype_weight,
        contrastive_weight=args.contrastive_weight,
        consistency_weight=args.consistency_weight
    )
    
    # Progressive training strategy
    training_strategy = ProgressiveTrainingStrategy(
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        freeze_epochs=args.freeze_epochs
    )
    
    # Prototype monitor
    prototype_monitor = PrototypeMonitor(args.num_prototypes) if args.monitor_prototypes else None
    
    # Training loop
    print("Starting improved pre-training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Update training strategy
        training_strategy.set_epoch(epoch)
        training_config = training_strategy.get_training_config()
        
        # Create optimizer with current configuration
        optimizer = create_improved_optimizer(model, args, training_config)
        
        # Train for one epoch
        train_metrics = train_epoch_improved(
            model, train_loader, criterion, optimizer, epoch, args,
            writer, training_strategy, prototype_monitor
        )
        
        # Log metrics
        print(f"Epoch {epoch}: Loss={train_metrics['total_loss']:.4f}, "
              f"La={train_metrics['align_loss']:.4f}, "
              f"Lp={train_metrics['prototype_loss']:.4f}, "
              f"Lc={train_metrics['contrastive_loss']:.4f}, "
              f"Phase={train_metrics['training_phase']}")
        
        if writer:
            writer.add_scalar('Epoch/Total_Loss', train_metrics['total_loss'], epoch)
            writer.add_scalar('Epoch/Align_Loss', train_metrics['align_loss'], epoch)
            writer.add_scalar('Epoch/Prototype_Loss', train_metrics['prototype_loss'], epoch)
            writer.add_scalar('Epoch/Contrastive_Loss', train_metrics['contrastive_loss'], epoch)
            writer.add_scalar('Epoch/Consistency_Loss', train_metrics['consistency_loss'], epoch)
        
        # Monitor prototype learning
        if prototype_monitor and epoch % args.debug_interval == 0:
            print(f"\n--- Prototype Analysis (Epoch {epoch}) ---")
            stats = prototype_monitor.update(model, test_loader, args.device, max_batches=5)
            if stats:
                prototype_monitor.print_summary()
                
                # Log prototype statistics
                if writer:
                    writer.add_scalar('Prototype/Sparsity', stats['sparsity'], epoch)
                    writer.add_scalar('Prototype/Diversity', stats['diversity'], epoch)
                    writer.add_scalar('Prototype/Active_Count', (stats['active_ratio'] > 0.1).sum().item(), epoch)
            
            # Debug prototype learning
            debug_prototype_learning(model, test_loader, args.device, epoch)
        
        # Save checkpoint
        is_best = train_metrics['total_loss'] < best_loss
        if is_best:
            best_loss = train_metrics['total_loss']
        
        if epoch % 20 == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'args': vars(args),
                'num_prototypes': args.num_prototypes,
                'backbone': args.backbone,
                'training_strategy': training_config
            }
            
            filename = 'checkpoint_best.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, filename))
            
            if is_best:
                print(f"★ New best loss: {best_loss:.4f}")
    
    # Final checkpoint and evaluation
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'train_metrics': train_metrics,
        'args': vars(args),
        'num_prototypes': args.num_prototypes,
        'backbone': args.backbone
    }
    
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'final_improved_model.pth'))
    
    # Final prototype analysis
    if prototype_monitor:
        print("\n" + "="*60)
        print("FINAL PROTOTYPE ANALYSIS")
        print("="*60)
        final_stats = prototype_monitor.update(model, test_loader, args.device, max_batches=10)
        if final_stats:
            prototype_monitor.print_summary()
            
            # Save prototype statistics
            with open(os.path.join(args.save_dir, 'prototype_stats.json'), 'w') as f:
                # Convert tensors to lists for JSON serialization
                stats_serializable = {}
                for k, v in final_stats.items():
                    if hasattr(v, 'tolist'):
                        stats_serializable[k] = v.tolist()
                    else:
                        stats_serializable[k] = v
                json.dump(stats_serializable, f, indent=2)
    
    print("Improved pre-training completed!")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()