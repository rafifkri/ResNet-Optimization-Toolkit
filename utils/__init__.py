"""
===================================================================================
Utils Package
===================================================================================
"""

from .dataloader import (
    get_loaders, get_loaders_from_config, get_calibration_loader, get_dataset_info
)

from .metrics import (
    count_parameters, get_model_complexity, measure_latency, get_model_size,
    accuracy, ClassificationMetrics, evaluate_model, compare_models, 
    print_comparison_table, get_sparsity
)

from .visualizer import (
    plot_training_curves, plot_lr_schedule, plot_confusion_matrix,
    plot_tsne, extract_features, plot_model_comparison, plot_pareto_front,
    plot_sparsity_analysis, plot_ablation_results
)

from .logger import (
    ExperimentLogger, AverageMeter, ProgressMeter, setup_seed
)

from .checkpoint import (
    CheckpointManager, EarlyStopping, save_checkpoint, load_checkpoint
)

from .scheduler import (
    WarmupCosineScheduler, WarmupMultiStepScheduler, create_scheduler, get_lr
)

from .augmentations import (
    Cutout, mixup_data, mixup_criterion, cutmix_data, 
    CIFAR10Policy, RandAugment, LabelSmoothingCrossEntropy,
    get_train_transforms, get_test_transforms
)

__all__ = [
    # Data
    'get_loaders', 'get_loaders_from_config', 'get_calibration_loader', 'get_dataset_info',
    # Metrics
    'count_parameters', 'get_model_complexity', 'measure_latency', 'get_model_size',
    'accuracy', 'ClassificationMetrics', 'evaluate_model', 'compare_models',
    'print_comparison_table', 'get_sparsity',
    # Visualization
    'plot_training_curves', 'plot_lr_schedule', 'plot_confusion_matrix',
    'plot_tsne', 'extract_features', 'plot_model_comparison', 'plot_pareto_front',
    'plot_sparsity_analysis', 'plot_ablation_results',
    # Logging
    'ExperimentLogger', 'AverageMeter', 'ProgressMeter', 'setup_seed',
    # Checkpoint
    'CheckpointManager', 'EarlyStopping', 'save_checkpoint', 'load_checkpoint',
    # Scheduler
    'WarmupCosineScheduler', 'WarmupMultiStepScheduler', 'create_scheduler', 'get_lr',
    # Augmentation
    'Cutout', 'mixup_data', 'mixup_criterion', 'cutmix_data',
    'CIFAR10Policy', 'RandAugment', 'LabelSmoothingCrossEntropy',
    'get_train_transforms', 'get_test_transforms',
]
