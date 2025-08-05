# Self-Supervised Pre-training of B-cos PIP-Net

This implementation combines B-cos Networks and PIP-Net for self-supervised pre-training of interpretable prototypes.

## Overview

The approach implements the following pipeline:

1. **6-channel Image Encoding**: Convert RGB images to [r,g,b,1-r,1-g,1-b] representation
2. **Contrastive Pairs**: Generate positive pairs through different data augmentations
3. **B-cos Backbone**: Use BcosConv2d layers instead of standard convolutions
4. **Loss Combination**: Pre-train using La (align_loss) + Lt (tanh_loss)
5. **Prototype Learning**: MaxPool2D to learn prototype presence indicators

## Architecture

- **Backbone**: ResNet18/50 with BcosConv2d layers accepting 6-channel input
- **Prototype Layer**: 1x1 convolution + softmax to generate prototype activations
- **Global Pooling**: AdaptiveMaxPool2d to get prototype presence scores
- **Projection Head**: For optional contrastive learning

## Files

- `transforms.py`: 6-channel encoding and contrastive augmentations
- `bcos_features.py`: B-cos enabled ResNet backbones
- `losses.py`: AlignLoss (La), TanhLoss (Lt), and contrastive losses
- `bcos_pipnet.py`: Main model architecture combining B-cos and PIP-Net
- `datasets.py`: Dataset setup for CIFAR-10 and CUB-200-2011
- `train_pretraining.py`: Main training script
- `run_pretraining.py`: Convenience script with predefined configurations

## Requirements

```bash
pip install torch torchvision tqdm tensorboard
```

## Usage

### Quick Start - CIFAR-10

```bash
cd src
python run_pretraining.py --experiment cifar10
```

### CUB-200-2011

First, prepare the CUB dataset:
```bash
# Download CUB-200-2011 and organize as train/test folders
# Or use the preprocessing script from PIPNet
python ../PIPNet/util/preprocess_cub.py --data_dir ./data/CUB_200_2011
```

Then run pre-training:
```bash
python run_pretraining.py --experiment cub --cub_data_dir ./data/CUB_200_2011
```

### Custom Training

```bash
python train_pretraining.py --dataset cifar10 \
                           --backbone bcos_resnet18 \
                           --num_prototypes 256 \
                           --epochs 200 \
                           --align_weight 1.0 \
                           --tanh_weight 1.0 \
                           --device cuda
```

## Hyperparameters

### CIFAR-10 (Recommended)
- **Backbone**: bcos_resnet18
- **Prototypes**: 256
- **Batch size**: 128
- **Learning rate**: 1e-3
- **Epochs**: 200
- **Loss weights**: La=1.0, Lt=1.0

### CUB-200-2011 (Recommended)
- **Backbone**: bcos_resnet18
- **Prototypes**: 512
- **Batch size**: 64
- **Learning rate**: 1e-3
- **Epochs**: 200
- **Loss weights**: La=1.0, Lt=1.0

## Loss Functions

### Align Loss (La)
Encourages diversity in B-cos filter weights by penalizing high cosine similarity between different filters.

### Tanh Loss (Lt)
Applied to prototype features to encourage sparse, binary-like activations:
```
Lt = mean((tanh(proto_features/temperature)^2 - 1)^2)
```

### Optional Contrastive Loss
InfoNCE loss on projected prototype features for additional self-supervised signal.

## Monitoring

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir logs/
```

Metrics include:
- Total loss (La + Lt + optional contrastive)
- Individual loss components
- Learning rate schedule

## Output

Checkpoints are saved to `./checkpoints/` containing:
- Model state dict
- Optimizer state dict
- Training metrics
- Model configuration

## GPU Requirements

- **CIFAR-10**: ~4GB VRAM (batch_size=128)
- **CUB-200-2011**: ~6GB VRAM (batch_size=64)

For limited VRAM, reduce batch size or use bcos_resnet18 instead of bcos_resnet50.

## Next Steps

After pre-training, the learned prototypes can be used for:
1. **Fine-tuning**: Add classification head and fine-tune on labeled data
2. **Analysis**: Visualize learned prototypes and their interpretability
3. **Transfer Learning**: Use as feature extractor for downstream tasks

## Integration with Original Codebases

The implementation is designed to be compatible with:
- **B-cos**: Uses the same BcosConv2d modules and loss concepts
- **PIP-Net**: Maintains the prototype-based architecture and can be fine-tuned using PIP-Net's training pipeline

## Citation

This implementation combines ideas from:
- B-cos Networks: [B-cos Networks: Alignment is All we Need for Interpretability](https://openaccess.thecvf.com/content/CVPR2022/html/Bohle_B-Cos_Networks_Alignment_Is_All_We_Need_for_Interpretability_CVPR_2022_paper.html)
- PIP-Net: [PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.pdf)