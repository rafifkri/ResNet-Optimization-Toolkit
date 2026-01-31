"""
===================================================================================
Configuration Module for ResNet Optimization Research
===================================================================================
Paper: Efficient ResNet Optimization via Ghost Modules, Coordinate Attention,
       Structured Pruning, Knowledge Distillation, and INT8 Quantization

Author: [Your Name]
Date: 2026
===================================================================================
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

# ===================================================================================
# ENUMS
# ===================================================================================

class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "imagenet"
    CUSTOM = "custom"

class ModelType(Enum):
    RESNET18_BASE = "resnet18_base"
    RESNET34_BASE = "resnet34_base"
    RESNET50_BASE = "resnet50_base"
    GHOST_RESNET18 = "ghost_resnet18"
    GHOST_RESNET34 = "ghost_resnet34"
    GHOST_RESNET50 = "ghost_resnet50"

class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"

class SchedulerType(Enum):
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    ONECYCLE = "onecycle"
    WARMUP_COSINE = "warmup_cosine"

class PruningType(Enum):
    UNSTRUCTURED_L1 = "unstructured_l1"
    STRUCTURED_LN = "structured_ln"
    STRUCTURED_CHANNEL = "structured_channel"
    GLOBAL_UNSTRUCTURED = "global_unstructured"

class QuantizationType(Enum):
    DYNAMIC = "dynamic"
    STATIC_PTQ = "static_ptq"
    QAT = "qat"  # Quantization-Aware Training

# ===================================================================================
# DATA CONFIGURATION
# ===================================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and augmentation"""
    dataset: DatasetType = DatasetType.CIFAR10
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1  # 10% untuk validasi
    
    # Augmentation settings
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    use_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 16
    use_autoaugment: bool = True
    use_randaugment: bool = False
    randaug_n: int = 2
    randaug_m: int = 9
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Image size (for different datasets)
    image_size: int = 32  # CIFAR default
    
    # Normalization (CIFAR-10 default)
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)

# ===================================================================================
# MODEL CONFIGURATION
# ===================================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_type: ModelType = ModelType.GHOST_RESNET18
    num_classes: int = 10
    
    # Ghost Module settings
    ghost_ratio: int = 2
    ghost_dw_size: int = 3
    
    # Attention settings
    use_attention: bool = True
    attention_type: str = "coordinate"  # "se", "cbam", "coordinate", "eca"
    attention_reduction: int = 32
    
    # Width multiplier (untuk model lebih kecil/besar)
    width_multiplier: float = 1.0
    
    # Dropout
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0  # Stochastic depth
    
    # Pretrained
    pretrained: bool = False
    pretrained_path: Optional[str] = None

# ===================================================================================
# TRAINING CONFIGURATION
# ===================================================================================

@dataclass
class TrainConfig:
    """Configuration for training"""
    epochs: int = 200
    
    # Optimizer
    optimizer: OptimizerType = OptimizerType.SGD
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    
    # For Adam/AdamW
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    
    # For StepLR
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [60, 120, 160])
    
    # Mixed Precision Training (AMP)
    use_amp: bool = True
    
    # Gradient clipping
    grad_clip: float = 1.0
    grad_clip_norm: bool = True
    
    # Checkpointing
    save_freq: int = 10
    save_best: bool = True
    resume: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 30
    min_delta: float = 0.001
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_interval: int = 50
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "resnet-optimization"
    experiment_name: str = "baseline"

# ===================================================================================
# PRUNING CONFIGURATION
# ===================================================================================

@dataclass
class PruneConfig:
    """Configuration for model pruning"""
    pruning_type: PruningType = PruningType.STRUCTURED_CHANNEL
    
    # Sparsity levels untuk eksperimen
    sparsity_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    target_sparsity: float = 0.5
    
    # Iterative pruning
    iterative_pruning: bool = True
    pruning_steps: int = 10
    
    # Fine-tuning setelah pruning
    finetune_epochs: int = 100
    finetune_lr: float = 1e-4
    
    # Global pruning
    global_pruning: bool = True
    
    # Layers to prune
    prune_conv: bool = True
    prune_linear: bool = False
    prune_bn: bool = False
    
    # Importance criteria
    importance_type: str = "l1"  # "l1", "l2", "taylor", "random"

# ===================================================================================
# DISTILLATION CONFIGURATION
# ===================================================================================

@dataclass
class DistillConfig:
    """Configuration for knowledge distillation"""
    # Teacher model
    teacher_model: ModelType = ModelType.RESNET50_BASE
    teacher_checkpoint: str = "weights/teacher_resnet50.pth"
    
    # Student model
    student_model: ModelType = ModelType.GHOST_RESNET18
    
    # Distillation parameters
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for soft loss vs hard loss
    
    # Feature distillation
    use_feature_distill: bool = True
    feature_layers: List[str] = field(default_factory=lambda: ["layer2", "layer3", "layer4"])
    feature_loss_weight: float = 0.1
    
    # Attention transfer
    use_attention_transfer: bool = True
    attention_loss_weight: float = 0.1
    
    # Progressive distillation
    progressive: bool = False
    progressive_stages: int = 3
    
    # Training
    epochs: int = 200
    learning_rate: float = 0.1

# ===================================================================================
# QUANTIZATION CONFIGURATION
# ===================================================================================

@dataclass
class QuantConfig:
    """Configuration for quantization"""
    quant_type: QuantizationType = QuantizationType.STATIC_PTQ
    
    # Backend: 'fbgemm' untuk x86 CPU, 'qnnpack' untuk ARM (mobile)
    backend: str = "fbgemm"
    
    # Calibration
    calibration_batches: int = 100
    
    # QAT settings
    qat_epochs: int = 10
    qat_lr: float = 0.0001
    
    # Observer type
    activation_observer: str = "histogram"  # "minmax", "histogram", "movingavg"
    weight_observer: str = "minmax"
    
    # Per-channel quantization (lebih akurat)
    per_channel: bool = True
    
    # Fuse modules
    fuse_modules: bool = True

# ===================================================================================
# DEPLOYMENT CONFIGURATION
# ===================================================================================

@dataclass
class DeployConfig:
    """Configuration for deployment and benchmarking"""
    # Export formats
    export_onnx: bool = True
    export_torchscript: bool = True
    export_tflite: bool = False
    
    # ONNX settings
    onnx_opset: int = 13
    onnx_simplify: bool = True
    
    # Benchmark settings
    benchmark_iterations: int = 100
    warmup_iterations: int = 20
    
    # Target devices
    benchmark_cpu: bool = True
    benchmark_gpu: bool = True
    
    # Input resolution untuk benchmark
    input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)

# ===================================================================================
# EXPERIMENT CONFIGURATION
# ===================================================================================

@dataclass
class ExperimentConfig:
    """Master configuration combining all configs"""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    
    # Paths
    output_dir: str = "./outputs"
    weights_dir: str = "./weights"
    logs_dir: str = "./logs"
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    def setup_dirs(self):
        """Create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)

# ===================================================================================
# PRESET CONFIGURATIONS (untuk paper)
# ===================================================================================

def get_cifar10_config() -> ExperimentConfig:
    """Configuration for CIFAR-10 experiments"""
    config = ExperimentConfig()
    config.data.dataset = DatasetType.CIFAR10
    config.data.image_size = 32
    config.model.num_classes = 10
    config.deploy.input_size = (1, 3, 32, 32)
    return config

def get_cifar100_config() -> ExperimentConfig:
    """Configuration for CIFAR-100 experiments"""
    config = ExperimentConfig()
    config.data.dataset = DatasetType.CIFAR100
    config.data.image_size = 32
    config.data.mean = (0.5071, 0.4867, 0.4408)
    config.data.std = (0.2675, 0.2565, 0.2761)
    config.model.num_classes = 100
    config.deploy.input_size = (1, 3, 32, 32)
    return config

def get_imagenet_config() -> ExperimentConfig:
    """Configuration for ImageNet experiments"""
    config = ExperimentConfig()
    config.data.dataset = DatasetType.IMAGENET
    config.data.image_size = 224
    config.data.batch_size = 256
    config.data.mean = (0.485, 0.456, 0.406)
    config.data.std = (0.229, 0.224, 0.225)
    config.model.num_classes = 1000
    config.train.epochs = 90
    config.train.milestones = [30, 60, 80]
    config.deploy.input_size = (1, 3, 224, 224)
    return config

def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """Generate configs for ablation study"""
    configs = {}
    
    # Base config
    base = get_cifar10_config()
    
    # 1. Without Ghost Module (standard ResNet)
    no_ghost = get_cifar10_config()
    no_ghost.model.model_type = ModelType.RESNET18_BASE
    no_ghost.train.experiment_name = "ablation_no_ghost"
    configs["no_ghost"] = no_ghost
    
    # 2. Without Attention
    no_attention = get_cifar10_config()
    no_attention.model.use_attention = False
    no_attention.train.experiment_name = "ablation_no_attention"
    configs["no_attention"] = no_attention
    
    # 3. Without Mixup/CutMix
    no_aug = get_cifar10_config()
    no_aug.data.use_mixup = False
    no_aug.data.use_cutmix = False
    no_aug.train.experiment_name = "ablation_no_mixup_cutmix"
    configs["no_mixup_cutmix"] = no_aug
    
    # 4. Different attention types
    for att_type in ["se", "cbam", "eca", "coordinate"]:
        att_config = get_cifar10_config()
        att_config.model.attention_type = att_type
        att_config.train.experiment_name = f"ablation_attention_{att_type}"
        configs[f"attention_{att_type}"] = att_config
    
    # 5. Different width multipliers
    for wm in [0.5, 0.75, 1.0, 1.25, 1.5]:
        wm_config = get_cifar10_config()
        wm_config.model.width_multiplier = wm
        wm_config.train.experiment_name = f"ablation_width_{wm}"
        configs[f"width_{wm}"] = wm_config
    
    # 6. Different sparsity levels
    for sp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        sp_config = get_cifar10_config()
        sp_config.prune.target_sparsity = sp
        sp_config.train.experiment_name = f"ablation_sparsity_{sp}"
        configs[f"sparsity_{sp}"] = sp_config
    
    return configs


if __name__ == "__main__":
    # Test configuration
    config = get_cifar10_config()
    print(f"Dataset: {config.data.dataset}")
    print(f"Model: {config.model.model_type}")
    print(f"Device: {config.get_device()}")
