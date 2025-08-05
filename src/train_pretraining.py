import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm
import json

from bcos_pipnet import create_bcos_pipnet
from losses import CombinedPretrainingLoss, InfoNCELoss
from datasets import get_dataset
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised pre-training of B-cos PIP-Net')
    
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
    parser.add_argument('--num_prototypes', type=int, default=512,
                        help='Number of prototypes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs')
    
    # Loss weights
    parser.add_argument('--align_weight', type=float, default=1.0,
                        help='Weight for alignment loss')
    parser.add_argument('--tanh_weight', type=float, default=1.0,
                        help='Weight for tanh loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='Weight for contrastive loss (optional)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Save directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save interval')
    
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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + torch.cos(torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_epoch(model, train_loader, criterion, contrastive_criterion, optimizer, epoch, args, writer):
    model.train()
    total_loss = 0.0
    total_align_loss = 0.0
    total_tanh_loss = 0.0
    total_contrastive_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (view1, view2, labels) in enumerate(pbar):
        view1, view2 = view1.to(args.device), view2.to(args.device)
        
        optimizer.zero_grad()
        
        # Forward pass through both views
        proto_features1, pooled1, projected1 = model(view1, return_features=True)
        proto_features2, pooled2, projected2 = model(view2, return_features=True)
        
        # Compute combined pre-training loss (La + Lt)
        loss_dict = criterion(model, proto_features1, proto_features2)
        loss = loss_dict['total_loss']
        
        # Add contrastive loss if specified
        contrastive_loss = torch.tensor(0.0, device=args.device)
        if args.contrastive_weight > 0:
            # Normalize projected features
            projected1_norm = F.normalize(projected1, dim=1)
            projected2_norm = F.normalize(projected2, dim=1)
            contrastive_loss = contrastive_criterion(projected1_norm, projected2_norm)
            loss += args.contrastive_weight * contrastive_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_align_loss += loss_dict['align_loss'].item()
        total_tanh_loss += loss_dict['tanh_loss'].item()
        total_contrastive_loss += contrastive_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'La': f'{loss_dict["align_loss"].item():.4f}',
            'Lt': f'{loss_dict["tanh_loss"].item():.4f}',
            'Lc': f'{contrastive_loss.item():.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % args.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/Align', loss_dict['align_loss'].item(), global_step)
            writer.add_scalar('Loss/Tanh', loss_dict['tanh_loss'].item(), global_step)
            if args.contrastive_weight > 0:
                writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), global_step)
    
    return {
        'total_loss': total_loss / len(train_loader),
        'align_loss': total_align_loss / len(train_loader),
        'tanh_loss': total_tanh_loss / len(train_loader),
        'contrastive_loss': total_contrastive_loss / len(train_loader)
    }


def save_checkpoint(model, optimizer, epoch, args, metrics, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args),
        'num_prototypes': args.num_prototypes,
        'backbone': args.backbone
    }
    
    filename = f'checkpoint_epoch_{epoch}.pth'
    if is_best:
        filename = 'checkpoint_best.pth'
    
    torch.save(checkpoint, os.path.join(args.save_dir, filename))


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
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create dataset
    print(f"Setting up {args.dataset} dataset...")
    if args.dataset == 'cifar10':
        dataset = get_dataset('cifar10', 
                            data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    else:  # cub
        dataset = get_dataset('cub',
                            data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            img_size=224)
    
    train_loader, test_loader = dataset.get_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.backbone} model with {args.num_prototypes} prototypes...")
    model = create_bcos_pipnet(
        num_prototypes=args.num_prototypes,
        backbone=args.backbone,
        pretrained=False
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss functions
    criterion = CombinedPretrainingLoss(
        align_weight=args.align_weight,
        tanh_weight=args.tanh_weight
    )
    
    contrastive_criterion = InfoNCELoss(temperature=args.temperature) if args.contrastive_weight > 0 else None
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, contrastive_criterion,
                                  optimizer, epoch, args, writer)
        
        # Log metrics
        print(f"Epoch {epoch}: Loss={train_metrics['total_loss']:.4f}, "
              f"La={train_metrics['align_loss']:.4f}, Lt={train_metrics['tanh_loss']:.4f}, "
              f"Lc={train_metrics['contrastive_loss']:.4f}, LR={lr:.6f}")
        
        if writer:
            writer.add_scalar('Learning_Rate', lr, epoch)
            writer.add_scalar('Epoch/Total_Loss', train_metrics['total_loss'], epoch)
            writer.add_scalar('Epoch/Align_Loss', train_metrics['align_loss'], epoch)
            writer.add_scalar('Epoch/Tanh_Loss', train_metrics['tanh_loss'], epoch)
            writer.add_scalar('Epoch/Contrastive_Loss', train_metrics['contrastive_loss'], epoch)
        
        # Save checkpoint
        is_best = train_metrics['total_loss'] < best_loss
        if is_best:
            best_loss = train_metrics['total_loss']
        
        if epoch % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, args, train_metrics, is_best=is_best)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, args.epochs - 1, args, train_metrics)
    
    print("Training completed!")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()