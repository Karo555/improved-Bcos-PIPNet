# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository combines two interpretable deep learning approaches:
- **B-cos Networks**: Models that achieve interpretability through alignment-based explanations
- **PIP-Net**: Patch-based Intuitive Prototypes for interpretable image classification

The project structure includes two main submodules:
- `B-cos/` - B-cos Networks implementation with experiments, interpretability analysis, and training utilities
- `PIPNet/` - PIP-Net implementation with prototype-based classification

## Development Commands

### Python Environment
- **Dependencies**: Install B-cos dependencies: `pip install -r B-cos/requirements.txt`
- **Project setup**: Project uses pyproject.toml but minimal dependencies defined
- **Python version**: Requires Python >=3.10

### Training Models

#### B-cos Networks
- Located in `B-cos/` directory
- Experiments configured in `B-cos/experiments/` with dataset-specific parameters
- Key modules: `B-cos/modules/bcosconv2d.py` for B-cos convolution layers

#### PIP-Net
- **Main training**: `python PIPNet/main.py [args]`
- **Help**: `python PIPNet/main.py --help` for all available arguments
- **Example datasets**: CUB-200-2011, CARS, PETS (see `PIPNet/used_arguments/` for parameters)
- **Key arguments**:
  - `--dataset`: Dataset name (CUB-200-2011, CARS, pets)
  - `--net`: Network backbone (convnext_tiny_26, resnet50)
  - `--epochs`: Training epochs
  - `--log_dir`: Output directory for logs and checkpoints

### Data Requirements
- Images should be organized in ImageFolder format: `root/class1/xxx.png`, `root/class2/yyy.png`
- Update dataset paths in `PIPNet/util/data.py`
- CUB preprocessing: Use `PIPNet/util/preprocess_cub.py`

### Visualization and Analysis
- **B-cos interpretability**: Jupyter notebooks in `B-cos/` root (Qualitative Examples.ipynb, Quantitative results.ipynb)
- **PIP-Net visualization**: Results saved to `--log_dir` including prototype visualizations and prediction explanations
- **Video evaluation**: `B-cos/VideoEvaluation.ipynb` for temporal stability analysis

## Architecture Overview

### B-cos Networks
- **Core concept**: Replaces standard convolutions with B-cos convolutions that enforce alignment between input and weight vectors
- **Key files**:
  - `B-cos/modules/bcosconv2d.py`: B-cos convolution implementation
  - `B-cos/models/bcos/`: B-cos versions of standard architectures (ResNet, DenseNet, VGG, Inception)
  - `B-cos/interpretability/`: Analysis tools for evaluating interpretability

### PIP-Net
- **Core concept**: Uses interpretable prototypes learned as patches from training images
- **Two-phase training**: 
  1. Prototype pretraining (`--epochs_pretrain`)
  2. Classification layer training with prototype freezing/unfreezing
- **Key components**:
  - `PIPNet/pipnet/pipnet.py`: Main PIP-Net architecture
  - `PIPNet/features/`: Feature extraction backbones (ResNet, ConvNext)
  - `PIPNet/util/`: Data loading, visualization, and utility functions

### Integration Points
- Both methods focus on interpretable computer vision
- Shared emphasis on prototype/patch-based explanations
- Compatible training pipelines for comparative analysis

## File Structure Notes

- `main.py` (root): Simple entry point, actual functionality in submodules
- `src/notebook.ipynb`: Development notebook (may contain experimental code)
- Git submodules: `B-cos/` and `PIPNet/` are separate repositories included as submodules
- Each submodule maintains its own README with detailed instructions