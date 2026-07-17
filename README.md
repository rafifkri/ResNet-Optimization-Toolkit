# Efficient ResNet Optimization for Edge Deployment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Paper Title**: *Efficient ResNet Optimization via Ghost Modules, Coordinate Attention, Structured Pruning, Knowledge Distillation, and INT8 Quantization for Resource-Constrained Devices*

## Abstract

This repository contains the implementation for our research on optimizing ResNet architectures for deployment on resource-constrained devices. We propose a comprehensive optimization pipeline combining:

1. **Ghost Modules** - Generating more features with fewer parameters
2. **Coordinate Attention** - Efficient spatial attention mechanism
3. **Structured Pruning** - Channel-wise pruning with fine-tuning
4. **Knowledge Distillation** - Transferring knowledge from larger teacher models
5. **INT8 Quantization** - Post-training and quantization-aware training

Our optimized Ghost-ResNet achieves **95.23% accuracy** with **5.77M parameters** (48% fewer than standard ResNet-18), **287M FLOPs**, and supports deployment on mobile devices via PyTorch Mobile.

## Project Structure

```
Optimasi_ResNet/
├── config.py                   # Central configuration module
├── main.py                     # Main entry point & CLI
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── resnet_base.py         # Baseline ResNet (18/34/50)
│   ├── resnet_ghost.py        # Ghost-ResNet variants
│   ├── model_factory.py       # Model builder utility
│   └── layers/
│       ├── __init__.py
│       ├── ghost_module.py    # Ghost Module implementation
│       ├── attention.py       # SE, CBAM, Coordinate, ECA attention
│       └── drop_path.py       # Stochastic depth
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── dataloader.py          # Data loading with augmentations
│   ├── metrics.py             # FLOPs, latency, accuracy metrics
│   ├── visualizer.py          # Training curves, confusion matrix, t-SNE
│   ├── logger.py              # Logging utilities
│   ├── checkpoint.py          # Model checkpointing
│   ├── augmentations.py       # Mixup, CutMix, Cutout, AutoAugment
│   └── scheduler.py           # Learning rate schedulers
│
├── experiments/                # Experiment scripts
│   ├── 01_baseline.py         # Train baseline ResNet
│   ├── 02_pruning.py          # Structured pruning experiments
│   ├── 03_distillation.py     # Knowledge distillation
│   ├── 04_quantization.py     # INT8 quantization
│   ├── 05_ablation.py         # Ablation studies
│   └── 06_comparison.py       # Full comparison & analysis
│
├── deployment/                 # Deployment utilities
│   ├── export_onnx.py         # ONNX export
│   ├── export_torchscript.py  # TorchScript export
│   ├── laptop_bench.py        # CPU benchmark
│   └── android_export.py      # Mobile deployment
│
├── data/                       # Dataset directory
│   ├── train/
│   ├── test/
│   └── meta/
│
├── weights/                    # Model checkpoints
│   ├── baseline/
│   ├── ghost/
│   ├── pruned/
│   ├── distilled/
│   └── quantized/
│
├── outputs/                    # Experiment outputs
│   ├── figures/               # Generated plots
│   ├── results/               # CSV/JSON results
│   └── logs/                  # Training logs
│
└── notebooks/                  # Jupyter notebooks (optional)
    └── analysis.ipynb
```

## System Requirements

- **Python** 3.8+
- **PyTorch** 2.0+
- **CUDA** 11.0+ (for GPU training)
- **GPU** NVIDIA GPU with 6GB+ VRAM recommended

Tested on:
- Windows 10/11
- NVIDIA GeForce RTX 3050 6GB Laptop GPU
- Intel Core (13th Gen)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Optimasi_ResNet.git
cd Optimasi_ResNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Baseline Model (GhostResNet-18 + Coordinate Attention)
```bash
python experiments/01_baseline.py
```

### 2. Apply Structured Pruning (50% sparsity)
```bash
python experiments/02_pruning.py
```

### 3. Knowledge Distillation
```bash
python experiments/03_distillation.py
```

### 4. INT8 Quantization
```bash
python experiments/04_quantization.py
```

### 5. Run Benchmark
```bash
python deployment/laptop_bench.py
```

### 6. Evaluate All Models
```bash
python evaluate_all.py
```

### 7. Generate Visualizations
```bash
python visualize_results.py
```

### 8. Export for Android
```bash
python deployment/android_export.py
```

## Results

### Main Results on CIFAR-10 (Evaluated on Test Set)

| Model | Params (M) | Accuracy (%) | Size (MB) | Notes |
|-------|------------|--------------|-----------|-------|
| **Baseline (GhostResNet-18 + CoordAtt)** | 5.77 | **95.23** | 22.05 | Best accuracy |
| Pruned (50% sparsity) | 5.77 | 93.00 | ~22 | -2.23% drop |
| Distilled (Student) | **1.46** | 93.74 | ~6 | 74% smaller |
| Quantized (Dynamic INT8) | 5.77 | 95.22 | 22.18 | Nearly lossless |

### Inference Benchmark (RTX 3050 Laptop GPU)

| Runtime | Batch Size | Latency (ms) | Throughput |
|---------|------------|--------------|------------|
| CPU | 1 | 102.93 | 9.7 samples/s |
| CPU | 32 | 699.48 | 45.7 samples/s |
| **GPU** | 1 | 60.07 | 16.6 samples/s |
| **GPU** | 32 | 51.91 | **616.4 samples/s** |

### Model Complexity

| Metric | Value |
|--------|-------|
| Parameters | 5.77M |
| FLOPs | 287.61M |
| MACs | 143.81M |

### Mobile Deployment

| Export Format | Size | Latency |
|---------------|------|---------|
| TorchScript (.pt) | 22.48 MB | 120.99 ms |
| PyTorch Mobile (.ptl) | 22.04 MB | 438.15 ms |

## Visualization

Training curves, confusion matrices, and comparison charts are generated in `weights/visualization/`:

- `accuracy_comparison.png` - Bar chart of model accuracies
- `params_comparison.png` - Parameter count comparison
- `accuracy_vs_params.png` - Trade-off scatter plot
- `inference_speed.png` - CPU vs GPU throughput
- `latency_comparison.png` - Latency per batch size
- `optimization_summary.png` - Summary table
- `compression_radar.png` - Radar chart comparison

## Output Files

```
weights/
├── baseline.pth                    # Trained baseline model
├── teacher.pth                     # Teacher model for distillation
├── evaluation_results.json         # Test set evaluation results
├── pruning_experiments/            # Pruning results
├── distillation_experiments/       # Distillation results
├── quantization_experiments/       # Quantized models (incl. TorchScript)
├── benchmark_results/              # Inference benchmark JSON
├── visualization/                  # Generated charts (7 PNG files)
└── mobile_export/                  # Android deployment files
    ├── ghost_resnet18.pt          # TorchScript model
    ├── ghost_resnet18_mobile.ptl  # PyTorch Mobile model
    └── android_code/              # Sample Kotlin code
```

## Methodology

### Ghost Module
Ghost modules generate more feature maps from cheap operations, reducing computational cost while maintaining representational capacity.

### Coordinate Attention
Unlike channel attention (SE) or spatial attention (CBAM), Coordinate Attention encodes channel relationships and long-range dependencies with precise positional information.

### Structured Pruning
We use L1-norm based filter pruning with iterative pruning schedule and fine-tuning to maintain accuracy.

### Knowledge Distillation
We employ both logit-based (soft targets) and feature-based distillation with attention transfer.

### INT8 Quantization
We use histogram-based calibration for post-training quantization and support quantization-aware training for minimal accuracy loss.

## Citation

If you find this work useful, please cite:

```bibtex
@article{yourname2026efficient,
  title={Efficient ResNet Optimization via Ghost Modules, Coordinate Attention, Structured Pruning, Knowledge Distillation, and INT8 Quantization},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```
## Acknowledgments

- Ghost Module: [GhostNet (CVPR 2020)](https://arxiv.org/abs/1911.11907)
- Coordinate Attention: [CA (CVPR 2021)](https://arxiv.org/abs/2103.02907)
- PyTorch Team for the excellent framework
