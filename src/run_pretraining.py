#!/usr/bin/env python3
"""
Convenience script to run self-supervised pre-training with different configurations
"""

import subprocess
import argparse
import os


def run_cifar10_pretraining():
    """Run pre-training on CIFAR-10 with optimal hyperparameters"""
    cmd = [
        'python', 'train_pretraining.py',
        '--dataset', 'cifar10',
        '--backbone', 'bcos_resnet18',
        '--num_prototypes', '256',
        '--batch_size', '128',
        '--epochs', '200',
        '--lr', '1e-3',
        '--align_weight', '1.0',
        '--tanh_weight', '1.0',
        '--warmup_epochs', '10',
        '--log_dir', './logs/cifar10_pretraining',
        '--save_dir', './checkpoints/cifar10_pretraining',
        '--device', 'cuda'
    ]
    
    print("Running CIFAR-10 pre-training...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def run_cub_pretraining(data_dir):
    """Run pre-training on CUB-200-2011 with optimal hyperparameters"""
    cmd = [
        'python', 'train_pretraining.py',
        '--dataset', 'cub',
        '--data_dir', data_dir,
        '--backbone', 'bcos_resnet18',
        '--num_prototypes', '512',
        '--batch_size', '64',
        '--epochs', '200',
        '--lr', '1e-3',
        '--align_weight', '1.0',
        '--tanh_weight', '1.0',
        '--warmup_epochs', '10',
        '--log_dir', './logs/cub_pretraining',
        '--save_dir', './checkpoints/cub_pretraining',
        '--device', 'cuda'
    ]
    
    print("Running CUB-200-2011 pre-training...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def run_cifar10_with_contrastive():
    """Run pre-training on CIFAR-10 with additional contrastive loss"""
    cmd = [
        'python', 'train_pretraining.py',
        '--dataset', 'cifar10',
        '--backbone', 'bcos_resnet18',
        '--num_prototypes', '256',
        '--batch_size', '128',
        '--epochs', '200',
        '--lr', '1e-3',
        '--align_weight', '1.0',
        '--tanh_weight', '1.0',
        '--contrastive_weight', '0.5',
        '--temperature', '0.07',
        '--warmup_epochs', '10',
        '--log_dir', './logs/cifar10_pretraining_contrastive',
        '--save_dir', './checkpoints/cifar10_pretraining_contrastive',
        '--device', 'cuda'
    ]
    
    print("Running CIFAR-10 pre-training with contrastive loss...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run self-supervised pre-training experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['cifar10', 'cub', 'cifar10_contrastive'],
                        help='Experiment to run')
    parser.add_argument('--cub_data_dir', type=str, default='./data/CUB_200_2011',
                        help='CUB dataset directory (required for CUB experiments)')
    
    args = parser.parse_args()
    
    # Change to src directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.experiment == 'cifar10':
        run_cifar10_pretraining()
    elif args.experiment == 'cub':
        if not os.path.exists(args.cub_data_dir):
            print(f"Error: CUB dataset not found at {args.cub_data_dir}")
            print("Please download and preprocess CUB-200-2011 dataset first.")
            return
        run_cub_pretraining(args.cub_data_dir)
    elif args.experiment == 'cifar10_contrastive':
        run_cifar10_with_contrastive()


if __name__ == '__main__':
    main()