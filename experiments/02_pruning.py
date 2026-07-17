import os
import sys
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_cifar10_config, PruningType, TrainConfig
from models import create_model
from utils.dataloader import get_loaders_from_config
from utils.metrics import get_model_complexity, measure_latency, get_sparsity, evaluate_model
from utils.logger import ExperimentLogger, AverageMeter
from utils.checkpoint import CheckpointManager
from utils.scheduler import create_scheduler as get_scheduler
from utils.augmentations import LabelSmoothingCrossEntropy


# ============================================================================
# Pruning Methods
# ============================================================================

class PruningMethod:
    """Base class for pruning methods."""
    
    @staticmethod
    def get_prunable_modules(model: nn.Module, 
                              module_types: Tuple = (nn.Conv2d, nn.Linear)) -> List[Tuple[str, nn.Module]]:
        """Get all prunable modules from model."""
        modules = []
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                modules.append((name, module))
        return modules


class UnstructuredPruning(PruningMethod):
    """Unstructured (weight-level) pruning methods."""
    
    @staticmethod
    def l1_unstructured(model: nn.Module, amount: float, 
                        module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Apply L1 unstructured pruning to all layers."""
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                prune.l1_unstructured(module, name='weight', amount=amount)
        return model
    
    @staticmethod
    def random_unstructured(model: nn.Module, amount: float,
                           module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Apply random unstructured pruning."""
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                prune.random_unstructured(module, name='weight', amount=amount)
        return model
    
    @staticmethod
    def global_unstructured(model: nn.Module, amount: float,
                           module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Apply global L1 unstructured pruning across all layers."""
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        return model


class StructuredPruning(PruningMethod):
    """Structured (filter/channel) pruning methods."""
    
    @staticmethod
    def ln_structured(model: nn.Module, amount: float, n: int = 2,
                      dim: int = 0, module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Apply Ln structured pruning (filter pruning when dim=0)."""
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                if module.weight.shape[dim] > 1:  # Skip if only 1 filter
                    prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
        return model
    
    @staticmethod
    def filter_pruning_by_norm(model: nn.Module, amount: float,
                               module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Prune filters with smallest L2 norm."""
        return StructuredPruning.ln_structured(model, amount, n=2, dim=0, module_types=module_types)
    
    @staticmethod
    def channel_pruning_by_norm(model: nn.Module, amount: float,
                                module_types: Tuple = (nn.Conv2d,)) -> nn.Module:
        """Prune input channels with smallest L2 norm."""
        return StructuredPruning.ln_structured(model, amount, n=2, dim=1, module_types=module_types)


class ImportanceBasedPruning(PruningMethod):
    """Importance-based pruning using activation statistics."""
    
    def __init__(self, model: nn.Module, dataloader, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.activation_means = {}
        self.hooks = []
    
    def _register_hooks(self):
        """Register forward hooks to collect activation statistics."""
        def hook_fn(name):
            def hook(module, input, output):
                if name not in self.activation_means:
                    self.activation_means[name] = []
                # Calculate mean activation per filter
                if len(output.shape) == 4:  # Conv output
                    mean_act = output.abs().mean(dim=(0, 2, 3)).detach().cpu()
                else:  # Linear output
                    mean_act = output.abs().mean(dim=0).detach().cpu()
                self.activation_means[name].append(mean_act)
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_importance(self, num_batches: int = 50) -> Dict[str, torch.Tensor]:
        """Compute filter importance based on activation statistics."""
        self.model.eval()
        self._register_hooks()
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        
        self._remove_hooks()
        
        # Aggregate importance scores
        importance = {}
        for name, acts in self.activation_means.items():
            importance[name] = torch.stack(acts).mean(dim=0)
        
        return importance
    
    def prune_by_importance(self, amount: float, importance: Dict[str, torch.Tensor] = None):
        """Prune filters based on computed importance."""
        if importance is None:
            importance = self.compute_importance()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in importance:
                imp = importance[name]
                num_filters = len(imp)
                num_prune = int(amount * num_filters)
                
                if num_prune > 0 and num_prune < num_filters:
                    # Create custom pruning mask based on importance
                    _, indices = torch.sort(imp)
                    mask = torch.ones(num_filters)
                    mask[indices[:num_prune]] = 0
                    
                    # Apply custom pruning
                    custom_mask = mask.view(-1, 1, 1, 1).expand_as(module.weight).to(module.weight.device)
                    prune.custom_from_mask(module, name='weight', mask=custom_mask)
        
        return self.model


# ============================================================================
# Sensitivity Analysis
# ============================================================================

def layer_sensitivity_analysis(model: nn.Module, 
                               dataloader,
                               device: str = 'cuda',
                               amounts: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                               criterion: nn.Module = None) -> Dict[str, Dict[float, float]]:
    """
    Analyze sensitivity of each layer to pruning.
    
    Returns accuracy drop for each layer at each pruning amount.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    
    # Get baseline accuracy
    baseline_acc = evaluate_single(model, dataloader, device)
    
    sensitivity = {}
    
    # Get all conv layers
    conv_layers = [(name, module) for name, module in model.named_modules() 
                   if isinstance(module, nn.Conv2d)]
    
    for layer_name, layer_module in tqdm(conv_layers, desc="Analyzing layers"):
        sensitivity[layer_name] = {}
        
        for amount in amounts:
            # Create a copy of the model
            model_copy = copy.deepcopy(model)
            
            # Find the same layer in the copy
            for name, module in model_copy.named_modules():
                if name == layer_name:
                    # Apply pruning to this layer only
                    if module.weight.shape[0] > 1:
                        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                    break
            
            # Evaluate
            acc = evaluate_single(model_copy, dataloader, device)
            sensitivity[layer_name][amount] = baseline_acc - acc
            
            del model_copy
            torch.cuda.empty_cache()
    
    return sensitivity


def evaluate_single(model: nn.Module, dataloader, device: str) -> float:
    """Quick evaluation to get accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


# ============================================================================
# Iterative Pruning with Fine-tuning
# ============================================================================

def iterative_pruning(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    target_sparsity: float = 0.9,
    num_iterations: int = 10,
    epochs_per_iteration: int = 10,
    learning_rate: float = 0.001,
    pruning_type: str = 'structured',
    use_amp: bool = True,
    logger: ExperimentLogger = None
) -> Tuple[nn.Module, Dict]:
    """
    Iterative pruning with fine-tuning (gradual pruning).
    
    This implements the gradual pruning schedule from:
    "To prune, or not to prune: exploring the efficacy of pruning for model compression"
    
    Args:
        model: Model to prune
        train_loader: Training data loader
        val_loader: Validation data loader  
        device: Device to use
        target_sparsity: Final target sparsity
        num_iterations: Number of pruning iterations
        epochs_per_iteration: Fine-tuning epochs after each pruning step
        learning_rate: Learning rate for fine-tuning
        pruning_type: 'structured' or 'unstructured'
        use_amp: Use automatic mixed precision
        logger: Experiment logger
    
    Returns:
        Pruned model and history dict
    """
    model = model.to(device)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_iteration * num_iterations)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    history = {
        'iteration': [],
        'sparsity': [],
        'accuracy': [],
        'params': [],
        'flops': []
    }
    
    # Initial evaluation
    initial_acc = evaluate_single(model, val_loader, device)
    initial_complexity = get_model_complexity(model)
    
    if logger:
        logger.info(f"Initial accuracy: {initial_acc:.2f}%")
        logger.info(f"Initial params: {initial_complexity['params_m']:.2f}M")
    
    history['iteration'].append(0)
    history['sparsity'].append(0.0)
    history['accuracy'].append(initial_acc)
    history['params'].append(initial_complexity['params_m'])
    history['flops'].append(initial_complexity['flops_m'])
    
    # Calculate pruning amount per iteration using polynomial schedule
    # sparsity_t = final_sparsity * (1 - (1 - t/n)^3)
    def get_sparsity_at_iteration(t, n, final_sparsity):
        return final_sparsity * (1 - (1 - t/n) ** 3)
    
    current_sparsity = 0
    
    for iteration in range(1, num_iterations + 1):
        target_sparsity_iter = get_sparsity_at_iteration(iteration, num_iterations, target_sparsity)
        
        # Calculate incremental pruning amount
        if current_sparsity < 1:
            incremental_amount = (target_sparsity_iter - current_sparsity) / (1 - current_sparsity)
        else:
            incremental_amount = 0
        
        if logger:
            logger.info(f"\n--- Iteration {iteration}/{num_iterations} ---")
            logger.info(f"Target sparsity: {target_sparsity_iter*100:.1f}%")
            logger.info(f"Incremental prune: {incremental_amount*100:.1f}%")
        
        # Apply pruning
        if incremental_amount > 0:
            if pruning_type == 'structured':
                StructuredPruning.filter_pruning_by_norm(model, incremental_amount)
            else:
                UnstructuredPruning.global_unstructured(model, incremental_amount)
        
        current_sparsity = get_sparsity(model)
        
        # Fine-tune
        model.train()
        for epoch in range(epochs_per_iteration):
            train_loss = AverageMeter('Loss')
            train_acc = AverageMeter('Acc')
            
            pbar = tqdm(train_loader, desc=f"Iter {iteration} Epoch {epoch+1}", leave=False)
            for inputs, targets in pbar:
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
                
                pbar.set_postfix({'loss': f'{train_loss.avg:.3f}', 'acc': f'{train_acc.avg:.1f}'})
            
            scheduler.step()
        
        # Evaluate
        val_acc = evaluate_single(model, val_loader, device)
        complexity = get_model_complexity(model)
        
        history['iteration'].append(iteration)
        history['sparsity'].append(current_sparsity)
        history['accuracy'].append(val_acc)
        history['params'].append(complexity['params_m'])
        history['flops'].append(complexity['flops_m'])
        
        if logger:
            logger.info(f"Sparsity: {current_sparsity*100:.1f}%")
            logger.info(f"Accuracy: {val_acc:.2f}%")
            logger.info(f"Params: {complexity['params_m']:.2f}M")
    
    return model, history


# ============================================================================
# Make Pruning Permanent
# ============================================================================

def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Remove pruning reparametrizations and make pruning permanent."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass  # No pruning mask on this module
            try:
                prune.remove(module, 'bias')
            except ValueError:
                pass
    return model


def remove_zero_filters(model: nn.Module) -> nn.Module:
    """
    Actually remove zero filters from structured pruning.
    This creates a physically smaller model.
    
    Note: This is complex for ResNet due to skip connections.
    For full implementation, consider using torch.fx or netadapt.
    """
    # This is a simplified version - full implementation would need
    # to handle residual connections carefully
    print("Warning: Full filter removal requires architectural changes.")
    print("The current model has zero masks but same shape.")
    return model


# ============================================================================
# Main Training Functions
# ============================================================================

def run_pruning_experiment(args):
    """Run complete pruning experiment."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get config
    config = get_cifar10_config()
    if args.dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    # Setup directories
    experiment_name = f"pruning_{args.pruning_type}_{args.target_sparsity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = ExperimentLogger(experiment_name, str(save_dir), use_tensorboard=False)  # Disabled - NumPy 2.x compatibility
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_loaders_from_config(config.data)
    
    # Get num_classes from dataset
    num_classes = 10 if args.dataset == 'cifar10' else 100 if args.dataset == 'cifar100' else 1000
    config.data.num_classes = num_classes
    
    # Load pretrained model
    model = create_model(
        args.model,
        num_classes=num_classes,
        attention_type=args.attention,
        pretrained=False
    )
    
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Loading pretrained weights from {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out profiling keys (total_ops, total_params from thop/ptflops)
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.endswith(('total_ops', 'total_params'))}
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded {len(state_dict)} parameters from checkpoint")
        else:
            print(f"Warning: Pretrained path {args.pretrained_path} not found. Using random init.")
    
    model = model.to(device)
    
    # Initial evaluation
    logger.info("\n" + "="*50)
    logger.info("Initial Model Analysis")
    logger.info("="*50)
    
    # Use val_loader if available, otherwise use test_loader
    eval_loader = val_loader if val_loader is not None else test_loader
    
    initial_acc = evaluate_single(model, eval_loader, device)
    initial_complexity = get_model_complexity(model)
    
    # Get params and flops with fallback to 0 if not available
    params_m = initial_complexity.get('params_m', initial_complexity.get('total_m', 0))
    flops_m = initial_complexity.get('flops_m', 0)
    
    logger.info(f"Accuracy: {initial_acc:.2f}%")
    logger.info(f"Parameters: {params_m:.2f}M")
    logger.info(f"FLOPs: {flops_m:.2f}M")
    
    # Run sensitivity analysis if requested
    if args.sensitivity_analysis:
        logger.info("\n" + "="*50)
        logger.info("Layer Sensitivity Analysis")
        logger.info("="*50)
        
        sensitivity = layer_sensitivity_analysis(
            model, val_loader, device,
            amounts=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # Save sensitivity results
        with open(save_dir / 'sensitivity_analysis.json', 'w') as f:
            json.dump({k: {str(kk): vv for kk, vv in v.items()} 
                      for k, v in sensitivity.items()}, f, indent=2)
        
        # Find most/least sensitive layers
        sensitivity_summary = {}
        for layer, amounts in sensitivity.items():
            sensitivity_summary[layer] = sum(amounts.values()) / len(amounts)
        
        sorted_layers = sorted(sensitivity_summary.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\nMost sensitive layers (highest accuracy drop):")
        for layer, drop in sorted_layers[:5]:
            logger.info(f"  {layer}: {drop:.2f}% avg drop")
        
        logger.info("\nLeast sensitive layers (lowest accuracy drop):")
        for layer, drop in sorted_layers[-5:]:
            logger.info(f"  {layer}: {drop:.2f}% avg drop")
    
    # Apply pruning
    logger.info("\n" + "="*50)
    logger.info(f"Applying {args.pruning_type} Pruning")
    logger.info("="*50)
    
    if args.iterative:
        # Iterative pruning with fine-tuning
        model, history = iterative_pruning(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            target_sparsity=args.target_sparsity,
            num_iterations=args.num_iterations,
            epochs_per_iteration=args.epochs_per_iteration,
            learning_rate=args.learning_rate,
            pruning_type=args.pruning_type,
            use_amp=args.amp,
            logger=logger
        )
        
        # Save history
        with open(save_dir / 'pruning_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    else:
        # One-shot pruning
        if args.pruning_type == 'structured':
            StructuredPruning.filter_pruning_by_norm(model, args.target_sparsity)
        elif args.pruning_type == 'unstructured':
            UnstructuredPruning.global_unstructured(model, args.target_sparsity)
        elif args.pruning_type == 'random':
            UnstructuredPruning.random_unstructured(model, args.target_sparsity)
        
        # Fine-tune if requested
        if args.finetune_epochs > 0:
            logger.info(f"\nFine-tuning for {args.finetune_epochs} epochs...")
            model = finetune_pruned_model(
                model, train_loader, val_loader, device,
                epochs=args.finetune_epochs,
                learning_rate=args.learning_rate,
                use_amp=args.amp,
                logger=logger
            )
    
    # Make pruning permanent
    make_pruning_permanent(model)
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Model Analysis")
    logger.info("="*50)
    
    final_acc = evaluate_single(model, test_loader, device)
    final_sparsity_dict = get_sparsity(model)
    final_sparsity = final_sparsity_dict['global_sparsity']
    final_complexity = get_model_complexity(model)
    
    # Measure latency
    latency_stats = measure_latency(model, (1, 3, 32, 32), device=str(device))
    
    logger.info(f"Test Accuracy: {final_acc:.2f}%")
    logger.info(f"Sparsity: {final_sparsity:.1f}%")
    logger.info(f"Parameters: {final_complexity['params_m']:.2f}M")
    logger.info(f"FLOPs: {final_complexity['flops_m']:.2f}M")
    logger.info(f"Latency: {latency_stats['mean_ms']:.2f} ± {latency_stats['std_ms']:.2f} ms")
    
    # Calculate compression ratio
    compression_ratio = initial_complexity['params_m'] / max(final_complexity['params_m'], 1e-6)
    speedup = initial_complexity['flops_m'] / max(final_complexity['flops_m'], 1e-6)
    
    logger.info(f"\nCompression ratio: {compression_ratio:.2f}x")
    logger.info(f"Theoretical speedup: {speedup:.2f}x")
    logger.info(f"Accuracy drop: {initial_acc - final_acc:.2f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_acc,
        'sparsity': final_sparsity,
        'compression_ratio': compression_ratio,
        'config': {
            'pruning_type': args.pruning_type,
            'target_sparsity': args.target_sparsity,
            'model': args.model
        }
    }, save_dir / 'pruned_model.pth')
    
    # Save summary
    summary = {
        'initial': {
            'accuracy': initial_acc,
            'params_m': initial_complexity['params_m'],
            'flops_m': initial_complexity['flops_m']
        },
        'final': {
            'accuracy': final_acc,
            'sparsity': final_sparsity,
            'params_m': final_complexity['params_m'],
            'flops_m': final_complexity['flops_m'],
            'latency_ms': latency_stats['mean_ms']
        },
        'compression': {
            'ratio': compression_ratio,
            'speedup': speedup,
            'accuracy_drop': initial_acc - final_acc
        }
    }
    
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {save_dir}")
    logger.close()
    
    return model, summary


def finetune_pruned_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    logger: ExperimentLogger = None
) -> nn.Module:
    """Fine-tune a pruned model."""
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = AverageMeter('Loss')
        train_acc = AverageMeter('Acc')
        
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
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
            
            pbar.set_postfix({'loss': f'{train_loss.avg:.3f}', 'acc': f'{train_acc.avg:.1f}'})
        
        scheduler.step()
        
        # Validate
        val_acc = evaluate_single(model, val_loader, device)
        
        if logger:
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss.avg:.3f}, "
                      f"Train Acc={train_acc.avg:.1f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


# ============================================================================
# Comparison Experiments
# ============================================================================

def compare_pruning_methods(args):
    """Compare different pruning methods."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get config
    config = get_cifar10_config()
    train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
    
    results = {}
    methods = ['structured', 'unstructured', 'random']
    sparsities = [0.3, 0.5, 0.7, 0.9]
    
    # Load base model
    base_model = create_model(args.model, num_classes=num_classes)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint)
    
    base_acc = evaluate_single(base_model.to(device), val_loader, device)
    
    for method in methods:
        results[method] = {}
        for sparsity in sparsities:
            print(f"\nTesting {method} pruning at {sparsity*100}% sparsity...")
            
            model = copy.deepcopy(base_model).to(device)
            
            if method == 'structured':
                StructuredPruning.filter_pruning_by_norm(model, sparsity)
            elif method == 'unstructured':
                UnstructuredPruning.global_unstructured(model, sparsity)
            else:
                UnstructuredPruning.random_unstructured(model, sparsity)
            
            acc = evaluate_single(model, val_loader, device)
            actual_sparsity = get_sparsity(model)
            
            results[method][sparsity] = {
                'accuracy': acc,
                'accuracy_drop': base_acc - acc,
                'actual_sparsity': actual_sparsity
            }
            
            print(f"  Accuracy: {acc:.2f}% (drop: {base_acc - acc:.2f}%)")
            
            del model
            torch.cuda.empty_cache()
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'pruning_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*70)
    print("Pruning Method Comparison")
    print("="*70)
    print(f"{'Method':<15} {'Sparsity':<10} {'Accuracy':<10} {'Drop':<10}")
    print("-"*70)
    
    for method in methods:
        for sparsity in sparsities:
            r = results[method][sparsity]
            print(f"{method:<15} {sparsity*100:<10.0f}% {r['accuracy']:<10.2f}% {r['accuracy_drop']:<10.2f}%")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet Pruning Experiment')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                'ghost_resnet18', 'ghost_resnet34', 'ghost_resnet_small'],
                        help='Model architecture')
    parser.add_argument('--attention', type=str, default='none',
                        choices=['none', 'se', 'cbam', 'eca', 'coord', 'coordinate', 'triplet'],
                        help='Attention mechanism')
    parser.add_argument('--pretrained_path', type=str, default='weights/baseline.pth',
                        help='Path to pretrained model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    
    # Pruning
    parser.add_argument('--pruning_type', type=str, default='structured',
                        choices=['structured', 'unstructured', 'random'],
                        help='Type of pruning')
    parser.add_argument('--target_sparsity', type=float, default=0.5,
                        help='Target sparsity (0.0-1.0)')
    
    # Iterative pruning
    parser.add_argument('--iterative', action='store_true',
                        help='Use iterative pruning with fine-tuning')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of pruning iterations')
    parser.add_argument('--epochs_per_iteration', type=int, default=5,
                        help='Fine-tuning epochs per iteration')
    
    # Fine-tuning (for one-shot)
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='Fine-tuning epochs after one-shot pruning')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    
    # Analysis
    parser.add_argument('--sensitivity_analysis', action='store_true',
                        help='Run layer sensitivity analysis')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare different pruning methods')
    
    # Training
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    
    # Save
    parser.add_argument('--save_dir', type=str, default='weights/pruning_experiments',
                        help='Directory to save results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.compare_methods:
        compare_pruning_methods(args)
    else:
        run_pruning_experiment(args)
