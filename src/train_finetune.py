import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
import json
from tqdm import tqdm
import numpy as np

from finetune_classifier import (
    create_scoring_sheet_classifier, 
    FineTuningLoss, 
    evaluate_model,
    compute_prototype_purity
)
from datasets import SixChannelDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning B-cos PIP-Net for classification')
    
    # Model arguments
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained B-cos PIP-Net checkpoint')
    parser.add_argument('--freeze_prototypes', action='store_true', default=True,
                        help='Freeze prototype learning components')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cub'],
                        help='Dataset to use for fine-tuning')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for CUB dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for classifier')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (if not frozen)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    
    # Loss weights
    parser.add_argument('--nll_weight', type=float, default=1.0,
                        help='Weight for negative log-likelihood loss')
    parser.add_argument('--l1_weight', type=float, default=0.0001,
                        help='Weight for L1 regularization on classifier')
    parser.add_argument('--orthogonal_weight', type=float, default=0.0,
                        help='Weight for orthogonal regularization')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./logs/finetune',
                        help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/finetune',
                        help='Save directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log interval')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval')
    
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


def get_dataloaders(args):
    """Create dataloaders for fine-tuning"""
    if args.dataset == 'cifar10':
        # CIFAR-10 transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=test_transform
        )
        
        num_classes = 10
        
    elif args.dataset == 'cub':
        # CUB-200-2011 transforms
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'train'), transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'test'), transform=test_transform
        )
        
        num_classes = len(train_dataset.classes)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Wrap with 6-channel transformation
    train_dataset = SixChannelDataset(train_dataset)
    test_dataset = SixChannelDataset(test_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine learning rate schedule with warmup"""
    if epoch < args.warmup_epochs:
        lr_mult = epoch / args.warmup_epochs
    else:
        lr_mult = 0.5 * (1. + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:  # Classifier parameters
            param_group['lr'] = args.lr * lr_mult
        else:  # Backbone parameters (if not frozen)
            param_group['lr'] = args.lr_backbone * lr_mult
    
    return args.lr * lr_mult


def train_epoch(model, train_loader, criterion, optimizer, epoch, args, writer):
    """Train for one epoch"""
    model.train()
    
    # Freeze prototype components if specified
    if args.freeze_prototypes:
        model.backbone.eval()
        model.prototype_layer.eval()
    
    total_loss = 0.0
    total_nll_loss = 0.0
    total_l1_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        proto_features, pooled_features, logits, class_scores = model(inputs)
        
        # Compute loss
        loss_dict = criterion(logits, targets, model.classifier.weight)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Apply PIPNet-style weight clamping for sparsity
        model.apply_weight_clamping()
        
        # Statistics
        total_loss += loss.item()
        total_nll_loss += loss_dict['nll_loss'].item()
        if 'l1_loss' in loss_dict:
            total_l1_loss += loss_dict['l1_loss'].item()
        
        predicted = torch.argmax(class_scores, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'NLL': f'{loss_dict["nll_loss"].item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
        
        # Log to tensorboard
        if writer and batch_idx % args.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/NLL_Loss', loss_dict['nll_loss'].item(), global_step)
            writer.add_scalar('Train/Accuracy', accuracy, global_step)
            if 'l1_loss' in loss_dict:
                writer.add_scalar('Train/L1_Loss', loss_dict['l1_loss'].item(), global_step)
    
    return {
        'loss': total_loss / len(train_loader),
        'nll_loss': total_nll_loss / len(train_loader),
        'l1_loss': total_l1_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def save_checkpoint(model, optimizer, epoch, args, metrics, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    
    filename = f'finetune_checkpoint_epoch_{epoch}.pth'
    if is_best:
        filename = 'finetune_checkpoint_best.pth'
    
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
    with open(os.path.join(args.save_dir, 'finetune_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create dataloaders
    print(f"Setting up {args.dataset} dataset...")
    train_loader, test_loader, num_classes = get_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"Loading pretrained model from {args.pretrained_path}")
    model = create_scoring_sheet_classifier(
        pretrained_path=args.pretrained_path,
        num_classes=num_classes,
        freeze_prototypes=args.freeze_prototypes
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    criterion = FineTuningLoss(
        nll_weight=args.nll_weight,
        l1_weight=args.l1_weight,
        orthogonal_weight=args.orthogonal_weight
    )
    
    # Create optimizer
    if args.freeze_prototypes:
        # Only optimize classifier
        optimizer = optim.AdamW(model.classifier.parameters(), 
                              lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Different learning rates for classifier and backbone
        optimizer = optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': args.lr},
            {'params': list(model.backbone.parameters()) + list(model.prototype_layer.parameters()), 
             'lr': args.lr_backbone}
        ], weight_decay=args.weight_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['metrics'].get('accuracy', 0.0)
    
    # Training loop
    print("Starting fine-tuning...")
    
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                  epoch, args, writer)
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            print("Evaluating model...")
            eval_metrics = evaluate_model(model, test_loader, args.device, num_classes)
            
            # Compute purity
            purity_metrics = compute_prototype_purity(model, test_loader, args.device, num_classes)
            
            # Get sparsity metrics
            sparsity_metrics = model.get_sparsity_metrics()
            
            # Log metrics
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.2f}%, "
                  f"Test Acc={eval_metrics['accuracy']:.2f}%, "
                  f"Purity={purity_metrics['mean_purity']:.3f}, "
                  f"Sparsity={sparsity_metrics['sparsity_ratio']:.3f}, "
                  f"Active Prototypes={sparsity_metrics['active_prototypes']}/{sparsity_metrics['total_prototypes']}")
            
            if writer:
                writer.add_scalar('Learning_Rate', lr, epoch)
                writer.add_scalar('Train/Epoch_Loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/Epoch_Accuracy', train_metrics['accuracy'], epoch)
                writer.add_scalar('Test/Accuracy', eval_metrics['accuracy'], epoch)
                writer.add_scalar('Test/Active_Prototypes_Per_Sample', 
                                eval_metrics['active_prototypes_per_sample'], epoch)
                writer.add_scalar('Test/Unused_Prototypes', 
                                eval_metrics['num_unused_prototypes'], epoch)
                writer.add_scalar('Test/Mean_Purity', purity_metrics['mean_purity'], epoch)
                writer.add_scalar('Test/High_Purity_Prototypes', 
                                purity_metrics['high_purity_prototypes'], epoch)
                writer.add_scalar('Sparsity/Sparsity_Ratio', sparsity_metrics['sparsity_ratio'], epoch)
                writer.add_scalar('Sparsity/Active_Prototypes', sparsity_metrics['active_prototypes'], epoch)
                writer.add_scalar('Sparsity/Active_Weights', sparsity_metrics['active_weights'], epoch)
            
            # Save checkpoint
            is_best = eval_metrics['accuracy'] > best_accuracy
            if is_best:
                best_accuracy = eval_metrics['accuracy']
                print(f"New best accuracy: {best_accuracy:.2f}%")
            
            all_metrics = {**train_metrics, **eval_metrics, **purity_metrics}
            save_checkpoint(model, optimizer, epoch, args, all_metrics, is_best=is_best)
        
        else:
            # Log training metrics only
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.2f}%, LR={lr:.6f}")
            
            if writer:
                writer.add_scalar('Learning_Rate', lr, epoch)
                writer.add_scalar('Train/Epoch_Loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/Epoch_Accuracy', train_metrics['accuracy'], epoch)
    
    # Final evaluation
    print("Running final evaluation...")
    final_eval_metrics = evaluate_model(model, test_loader, args.device, num_classes)
    final_purity_metrics = compute_prototype_purity(model, test_loader, args.device, num_classes)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {final_eval_metrics['accuracy']:.2f}%")
    print(f"  Mean Prototype Purity: {final_purity_metrics['mean_purity']:.3f}")
    print(f"  High Purity Prototypes (>0.5): {final_purity_metrics['high_purity_prototypes']}")
    print(f"  Unused Prototypes: {final_eval_metrics['num_unused_prototypes']}")
    print(f"  Active Prototypes per Sample: {final_eval_metrics['active_prototypes_per_sample']:.1f}")
    
    # Save final results
    final_results = {
        'final_test_accuracy': final_eval_metrics['accuracy'],
        'final_purity': final_purity_metrics['mean_purity'],
        'training_args': vars(args),
        'eval_metrics': final_eval_metrics,
        'purity_metrics': final_purity_metrics
    }
    
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for k, v in final_results.items():
            if isinstance(v, dict):
                results_to_save[k] = {}
                for k2, v2 in v.items():
                    if hasattr(v2, 'tolist'):
                        results_to_save[k][k2] = v2.tolist()
                    else:
                        results_to_save[k][k2] = v2
            else:
                results_to_save[k] = v
        json.dump(results_to_save, f, indent=2)
    
    print("Fine-tuning completed!")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()