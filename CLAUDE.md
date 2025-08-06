# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements an integrated interpretable deep learning system that combines **B-cos Networks** and **PIP-Net**. The key innovation is using B-cos convolutions as the backbone for prototype-based learning, creating a fully interpretable pipeline from feature extraction to final predictions.

**Core Integration**: B-cos feature extractors → Prototype learning → Classification with interpretable reasoning

The project includes:
- `B-cos/` and `PIPNet/` - Original implementations as git submodules  
- `src/` - Novel integration combining both approaches
- Training notebooks demonstrating the integrated workflow

## Development Commands

### Environment Setup
- **Python**: Requires Python ≥3.10
- **Dependencies**: `pip install -r B-cos/requirements.txt` (torch, torchvision, captum, etc.)
- **GPU**: CUDA recommended for training (models work on CPU but training is slow)
- **Submodules**: `git submodule update --init --recursive` to initialize B-cos and PIPNet

### Core Training Workflows

#### 1. Self-Supervised Pre-training (Recommended First Step)
```bash
# Basic pre-training on CIFAR-10
python src/train_pretraining.py --dataset cifar10 --epochs 100 --backbone bcos_simple

# Improved pre-training with progressive strategy
python src/train_improved_pretraining.py --dataset cifar10 --epochs 200 \
  --num_prototypes 256 --backbone bcos_simple --progressive_training --adaptive_augmentation

# CUB-200 pre-training with larger backbone
python src/train_pretraining.py --dataset cub --data_dir ./data/CUB_200_2011 \
  --epochs 150 --backbone bcos_large --batch_size 64
```

#### 2. Fine-tuning for Classification
```bash
# Fine-tune pre-trained model for classification
python src/train_finetune.py --pretrained_path ./checkpoints/pretraining/final_model.pth \
  --dataset cifar10 --num_classes 10 --epochs 50 --freeze_backbone
```

#### 3. Standalone Training (Skip Pre-training)
```bash
# Train end-to-end without pre-training
python src/finetune_classifier.py --dataset cifar10 --epochs 100 \
  --backbone bcos_simple --num_prototypes 256
```

### Notebook-Based Training
- **BcosPIPNet_Training.ipynb**: Complete training pipeline with visualizations
- **BcosPIPNet_FineTuning.ipynb**: Fine-tuning pre-trained models  
- **BcosPIPNet_Visualizations.ipynb**: Prototype analysis and interpretability

### Key Parameters
- `--backbone`: bcos_simple, bcos_medium, bcos_large (Simple B-cos CNN backbones)
  - bcos_resnet18, bcos_resnet50 (backward compatibility - now use simple CNNs)
- `--num_prototypes`: 256-512 recommended (fewer = more interpretable)
- `--progressive_training`: Use progressive training strategy (warmup → prototype learning → joint)
- `--adaptive_augmentation`: Enhanced augmentation for better prototype diversity

## Architecture Details

### Integrated B-cos PIP-Net (src/bcos_pipnet.py)

**Two-Stage Architecture**:
1. **BcosPIPNet**: Self-supervised pre-training model
   - 6-channel input encoding: [r,g,b,1-r,1-g,1-b] for richer representation
   - B-cos backbone extracts interpretable features
   - Prototype projection layer maps features to prototype space
   - Contrastive projection head for self-supervised learning

2. **BcosPIPNetClassifier**: Classification model built from pre-trained components
   - Frozen B-cos backbone and prototype layers
   - Non-negative linear classifier (preserves interpretability)
   - Prototype-based predictions with explainable reasoning

### Critical B-cos Implementation Requirements
**IMPORTANT**: This implementation follows strict B-cos principles:
- ❌ **NO ReLU/BatchNorm/MaxPool** - These break B-cos interpretability  
- ✅ **MaxOut in BcosConv2d** - Built-in max_out=2 for feature selection
- ✅ **Cosine similarity** - All convolutions use alignment-based operations
- ✅ **Global average pooling** - Replaces spatial pooling operations
- ✅ **Linear projections** - No non-linear activations in critical paths

### Progressive Training Strategy (src/improved_bcos_pipnet.py)
1. **Warmup Phase** (epochs 0-20): Initialize prototypes with frozen backbone
2. **Prototype Learning** (epochs 20-60): Learn prototypes with frozen backbone  
3. **Joint Training** (epochs 60+): End-to-end training with dynamic loss weights

### Data Pipeline
- **6-channel encoding**: `src/transforms.py` - Enhanced input representation
- **Contrastive pairs**: Data augmentation creates view pairs for self-supervised learning
- **Progressive augmentation**: Intensity increases during training phases
- **Dataset handlers**: `src/datasets.py` supports CIFAR-10, CUB-200, custom datasets

### Loss Functions (src/losses.py, src/improved_losses.py)
- **Alignment Loss (La)**: Enforces B-cos alignment properties
- **Prototype Loss (Lt)**: Encourages sparse, diverse prototype activations
- **Contrastive Loss (Lc)**: Self-supervised learning objective
- **Consistency Loss**: Maintains consistency between augmented views

## Key Implementation Files

### Core Integration
- `src/bcos_pipnet.py`: Main model architectures combining B-cos + PIP-Net
- `src/improved_bcos_pipnet.py`: Enhanced version with progressive training
- `src/bcos_features.py`: B-cos feature extractors adapted for 6-channel input

### Training Scripts  
- `src/train_pretraining.py`: Basic self-supervised pre-training
- `src/train_improved_pretraining.py`: Advanced pre-training with progressive strategy
- `src/train_finetune.py`: Fine-tuning pre-trained models for classification
- `src/finetune_classifier.py`: End-to-end classification training

### Support Modules
- `src/transforms.py`: 6-channel encoding and contrastive augmentations
- `src/losses.py`: Combined loss functions for interpretable learning
- `src/datasets.py`: Dataset handlers with 6-channel support
- `src/visualizations.py`: Prototype analysis and interpretability tools

## Training Best Practices

### Recommended Workflow
1. **Pre-train** with self-supervised objectives (100-200 epochs)
2. **Fine-tune** for classification with frozen backbone (50-100 epochs)
3. **Analyze** learned prototypes for interpretability validation
4. **Optional**: End-to-end fine-tuning with reduced learning rate

### Model Configuration
- **Prototypes**: Start with 256, increase to 512 for complex datasets
- **Backbone**: bcos_simple for speed, bcos_medium/bcos_large for accuracy
- **Learning rates**: 1e-3 for prototypes, 1e-4 for backbone (differential)
- **Progressive training**: Highly recommended for stable prototype learning

### Memory and Performance
- **Batch size**: 64-128 depending on GPU memory (6-channel input uses more memory)
- **Workers**: 2-4 for data loading (6-channel transform is CPU-intensive)
- **Checkpointing**: Models save every 20 epochs and on improvement
- **Monitoring**: Built-in prototype diversity and sparsity tracking

## Visualization and Analysis
- **Training notebooks**: Complete workflows with real-time visualization
- **Prototype analysis**: Activation patterns, diversity metrics, interpretability scores
- **B-cos interpretability**: Alignment visualizations and feature attributions  
- **Integration analysis**: How B-cos features map to learned prototypes

## Integration Notes
- Both B-cos and PIPNet are included as git submodules (separate repos)
- `src/` contains novel integration - the main contribution of this repository
- Training can use either integrated models or original implementations for comparison
- Submodule READMEs provide additional details for individual components