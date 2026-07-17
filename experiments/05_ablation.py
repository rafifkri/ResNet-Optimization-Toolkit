"""
05_ablation.py - Comprehensive Ablation Study Framework

This script provides a systematic ablation study framework for:
1. Attention Mechanism Comparison
2. Width Multiplier Analysis
3. Depth vs Width Trade-off
4. Augmentation Strategy Comparison
5. Learning Rate Schedule Analysis
6. Regularization Effects
7. Ghost Module Analysis
8. Drop Path Rate Study

Reference:
- Ablation Studies in Machine Learning (https://arxiv.org/abs/2103.04906)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_cifar10_config
from models import create_model
from utils.dataloader import get_loaders_from_config
from utils.metrics import get_model_complexity, measure_latency
from utils.logger import ExperimentLogger, AverageMeter
from utils.checkpoint import CheckpointManager
from utils.augmentations import LabelSmoothingCrossEntropy
from utils.visualizer import plot_ablation_results, plot_training_curves


# ============================================================================
# Quick Training Loop for Ablation
# ============================================================================

def quick_train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    epochs: int = 50,
    learning_rate: float = 0.1,
    optimizer_type: str = 'sgd',
    scheduler_type: str = 'cosine',
    use_amp: bool = True,
    label_smoothing: float = 0.1,
    weight_decay: float = 5e-4
) -> Tuple[float, Dict]:
    """
    Quick training loop for ablation studies.
    
    Returns best validation accuracy and training history.
    """
    model = model.to(device)
    
    # Loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    # Optimizer
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.01,
                               weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.01,
                              weight_decay=weight_decay)
    
    # Scheduler
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    elif scheduler_type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs//2, 3*epochs//4], gamma=0.1
        )
    else:
        scheduler = None
    
    scaler = GradScaler('cuda', enabled=use_amp)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = AverageMeter('Loss')
        train_acc = AverageMeter('Acc')
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc = (outputs.argmax(1) == targets).float().mean().item() * 100
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc, inputs.size(0))
        
        if scheduler:
            scheduler.step()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    return best_acc, history


# ============================================================================
# Ablation Study: Attention Mechanisms
# ============================================================================

def ablation_attention(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/attention',
    device: str = 'cuda'
) -> Dict:
    """
    Compare different attention mechanisms.
    """
    print("\n" + "="*60)
    print("Ablation Study: Attention Mechanisms")
    print("="*60)
    
    attention_types = ['none', 'se', 'cbam', 'eca', 'coord', 'triplet']
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for attention in attention_types:
        print(f"\nTesting attention: {attention}")
        
        model = create_model(base_model, num_classes=num_classes, attention=attention)
        complexity = get_model_complexity(model)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, epochs=epochs
        )
        
        results[attention] = {
            'accuracy': best_acc,
            'params_m': complexity['params_m'],
            'flops_m': complexity['flops_m'],
            'history': history
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        print(f"  Params: {complexity['params_m']:.2f}M")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'attention_ablation.json'), 'w') as f:
        # Remove history for JSON (too large)
        save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                       for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    # Plot results
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    plot_ablation_results(names, accuracies, 
                         title='Attention Mechanism Comparison',
                         save_path=os.path.join(save_dir, 'attention_comparison.png'))
    
    return results


# ============================================================================
# Ablation Study: Width Multiplier
# ============================================================================

def ablation_width(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/width',
    device: str = 'cuda'
) -> Dict:
    """
    Analyze effect of width multiplier on accuracy and efficiency.
    """
    print("\n" + "="*60)
    print("Ablation Study: Width Multiplier")
    print("="*60)
    
    width_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for width in width_multipliers:
        print(f"\nTesting width_multiplier: {width}")
        
        model = create_model(base_model, num_classes=num_classes, 
                           width_multiplier=width)
        complexity = get_model_complexity(model)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, epochs=epochs
        )
        
        results[f'width_{width}'] = {
            'width_multiplier': width,
            'accuracy': best_acc,
            'params_m': complexity['params_m'],
            'flops_m': complexity['flops_m']
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        print(f"  Params: {complexity['params_m']:.2f}M")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'width_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Ablation Study: Model Depth
# ============================================================================

def ablation_depth(
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/depth',
    device: str = 'cuda'
) -> Dict:
    """
    Compare different model depths.
    """
    print("\n" + "="*60)
    print("Ablation Study: Model Depth")
    print("="*60)
    
    models_to_test = [
        ('resnet18', 'ResNet-18'),
        ('resnet34', 'ResNet-34'),
        ('resnet50', 'ResNet-50'),
        ('ghost_resnet18', 'Ghost-ResNet-18'),
        ('ghost_resnet_small', 'Ghost-ResNet-Small'),
        ('ghost_resnet_tiny', 'Ghost-ResNet-Tiny')
    ]
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for model_name, display_name in models_to_test:
        print(f"\nTesting model: {display_name}")
        
        try:
            model = create_model(model_name, num_classes=num_classes)
            complexity = get_model_complexity(model)
            
            best_acc, history = quick_train(
                model, train_loader, val_loader, device, epochs=epochs
            )
            
            results[model_name] = {
                'display_name': display_name,
                'accuracy': best_acc,
                'params_m': complexity['params_m'],
                'flops_m': complexity['flops_m']
            }
            
            print(f"  Accuracy: {best_acc:.2f}%")
            print(f"  Params: {complexity['params_m']:.2f}M")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {'error': str(e)}
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'depth_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Ablation Study: Augmentation Strategies
# ============================================================================

def ablation_augmentation(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/augmentation',
    device: str = 'cuda'
) -> Dict:
    """
    Compare different augmentation strategies.
    """
    print("\n" + "="*60)
    print("Ablation Study: Augmentation Strategies")
    print("="*60)
    
    from torchvision import transforms
    from utils.augmentations import Cutout, CIFAR10Policy, RandAugment
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    num_classes = config.data.num_classes
    
    # Define augmentation strategies
    augmentation_configs = {
        'baseline': {
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        },
        'cutout': {
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16)
            ])
        },
        'autoaugment': {
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        },
        'randaugment': {
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        },
        'autoaug_cutout': {
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16)
            ])
        }
    }
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    results = {}
    
    for aug_name, aug_config in augmentation_configs.items():
        print(f"\nTesting augmentation: {aug_name}")
        
        # Create custom dataloaders
        from torchvision import datasets
        from torch.utils.data import DataLoader
        
        data_dir = config.data.data_dir
        
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=data_dir, train=True, download=True,
                transform=aug_config['train_transform']
            )
            val_dataset = datasets.CIFAR10(
                root=data_dir, train=False, download=True,
                transform=val_transform
            )
        else:
            train_dataset = datasets.CIFAR100(
                root=data_dir, train=True, download=True,
                transform=aug_config['train_transform']
            )
            val_dataset = datasets.CIFAR100(
                root=data_dir, train=False, download=True,
                transform=val_transform
            )
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                 num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                               num_workers=4, pin_memory=True)
        
        model = create_model(base_model, num_classes=num_classes)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, epochs=epochs
        )
        
        results[aug_name] = {
            'accuracy': best_acc,
            'history': {
                'train_acc': history['train_acc'][-10:],  # Last 10 epochs
                'val_acc': history['val_acc'][-10:]
            }
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'augmentation_ablation.json'), 'w') as f:
        save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                       for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# Ablation Study: Learning Rate Schedules
# ============================================================================

def ablation_lr_schedule(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/lr_schedule',
    device: str = 'cuda'
) -> Dict:
    """
    Compare different learning rate schedules.
    """
    print("\n" + "="*60)
    print("Ablation Study: Learning Rate Schedules")
    print("="*60)
    
    schedules = ['cosine', 'step', 'multistep', 'none']
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, _, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for schedule in schedules:
        print(f"\nTesting LR schedule: {schedule}")
        
        model = create_model(base_model, num_classes=num_classes)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, 
            epochs=epochs,
            scheduler_type=schedule if schedule != 'none' else None
        )
        
        results[schedule] = {
            'accuracy': best_acc,
            'lr_curve': history['lr']
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'lr_schedule_ablation.json'), 'w') as f:
        save_results = {k: {'accuracy': v['accuracy']} for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# Ablation Study: Drop Path Rate
# ============================================================================

def ablation_drop_path(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/drop_path',
    device: str = 'cuda'
) -> Dict:
    """
    Analyze effect of drop path (stochastic depth) rate.
    """
    print("\n" + "="*60)
    print("Ablation Study: Drop Path Rate")
    print("="*60)
    
    drop_path_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, _, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for drop_rate in drop_path_rates:
        print(f"\nTesting drop_path_rate: {drop_rate}")
        
        model = create_model(base_model, num_classes=num_classes, 
                           drop_path_rate=drop_rate)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, epochs=epochs
        )
        
        results[f'drop_{drop_rate}'] = {
            'drop_path_rate': drop_rate,
            'accuracy': best_acc
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'drop_path_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Ablation Study: Label Smoothing
# ============================================================================

def ablation_label_smoothing(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 100,
    save_dir: str = 'weights/ablation/label_smoothing',
    device: str = 'cuda'
) -> Dict:
    """
    Analyze effect of label smoothing.
    """
    print("\n" + "="*60)
    print("Ablation Study: Label Smoothing")
    print("="*60)
    
    smoothing_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    config = get_cifar10_config()
    if dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, _, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    
    for smoothing in smoothing_values:
        print(f"\nTesting label_smoothing: {smoothing}")
        
        model = create_model(base_model, num_classes=num_classes)
        
        best_acc, history = quick_train(
            model, train_loader, val_loader, device, 
            epochs=epochs,
            label_smoothing=smoothing
        )
        
        results[f'smooth_{smoothing}'] = {
            'label_smoothing': smoothing,
            'accuracy': best_acc
        }
        
        print(f"  Accuracy: {best_acc:.2f}%")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'label_smoothing_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Comprehensive Ablation Suite
# ============================================================================

def run_all_ablations(
    base_model: str = 'resnet18',
    dataset: str = 'cifar10',
    epochs: int = 50,
    save_dir: str = 'weights/ablation',
    device: str = 'cuda',
    studies: List[str] = None
) -> Dict:
    """
    Run all ablation studies.
    """
    all_results = {}
    
    available_studies = {
        'attention': ablation_attention,
        'width': ablation_width,
        'depth': ablation_depth,
        'augmentation': ablation_augmentation,
        'lr_schedule': ablation_lr_schedule,
        'drop_path': ablation_drop_path,
        'label_smoothing': ablation_label_smoothing
    }
    
    if studies is None:
        studies = list(available_studies.keys())
    
    for study_name in studies:
        if study_name in available_studies:
            print(f"\n{'#'*60}")
            print(f"Running {study_name} ablation...")
            print(f"{'#'*60}")
            
            study_save_dir = os.path.join(save_dir, study_name)
            
            try:
                if study_name in ['depth']:
                    # Depth study doesn't use base_model
                    results = available_studies[study_name](
                        dataset=dataset, epochs=epochs, 
                        save_dir=study_save_dir, device=device
                    )
                else:
                    results = available_studies[study_name](
                        base_model=base_model, dataset=dataset, 
                        epochs=epochs, save_dir=study_save_dir, device=device
                    )
                all_results[study_name] = results
            except Exception as e:
                print(f"Error in {study_name} ablation: {e}")
                all_results[study_name] = {'error': str(e)}
    
    # Save combined results
    os.makedirs(save_dir, exist_ok=True)
    
    # Create summary
    summary = {}
    for study_name, results in all_results.items():
        if 'error' not in results:
            best_config = max(results.items(), 
                            key=lambda x: x[1].get('accuracy', 0) if isinstance(x[1], dict) else 0)
            summary[study_name] = {
                'best_config': best_config[0],
                'best_accuracy': best_config[1].get('accuracy', 0) if isinstance(best_config[1], dict) else 0
            }
    
    with open(os.path.join(save_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    for study_name, result in summary.items():
        print(f"{study_name}: Best config = {result['best_config']}, "
              f"Accuracy = {result['best_accuracy']:.2f}%")
    
    return all_results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Studies')
    
    parser.add_argument('--study', type=str, default='all',
                       choices=['attention', 'width', 'depth', 'augmentation',
                               'lr_schedule', 'drop_path', 'label_smoothing', 'all'],
                       help='Ablation study to run')
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Base model for ablation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per experiment')
    parser.add_argument('--save_dir', type=str, default='weights/ablation',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.study == 'all':
        run_all_ablations(
            base_model=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            save_dir=args.save_dir,
            device=args.device
        )
    else:
        # Run single study
        study_functions = {
            'attention': ablation_attention,
            'width': ablation_width,
            'depth': ablation_depth,
            'augmentation': ablation_augmentation,
            'lr_schedule': ablation_lr_schedule,
            'drop_path': ablation_drop_path,
            'label_smoothing': ablation_label_smoothing
        }
        
        if args.study in ['depth']:
            study_functions[args.study](
                dataset=args.dataset,
                epochs=args.epochs,
                save_dir=os.path.join(args.save_dir, args.study),
                device=args.device
            )
        else:
            study_functions[args.study](
                base_model=args.model,
                dataset=args.dataset,
                epochs=args.epochs,
                save_dir=os.path.join(args.save_dir, args.study),
                device=args.device
            )
