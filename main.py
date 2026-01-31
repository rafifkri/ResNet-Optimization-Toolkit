#!/usr/bin/env python3
"""
main.py - ResNet Optimization Research Orchestrator

This is the main entry point for running all experiments in the
ResNet optimization research project. It provides a unified CLI
for training, evaluation, optimization, and deployment.

Usage:
    python main.py train --model resnet18 --dataset cifar10
    python main.py prune --model resnet18 --sparsity 0.5
    python main.py distill --teacher resnet50 --student ghost_resnet18
    python main.py quantize --model resnet18 --type ptq
    python main.py benchmark --model resnet18 --compare all
    python main.py ablation --study attention
    python main.py export --model resnet18 --format onnx
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    get_cifar10_config, get_ablation_configs,
    DatasetType, ModelType, OptimizerType, SchedulerType,
    PruningType, QuantizationType
)
from models import create_model, get_available_models
from utils.dataloader import get_loaders_from_config
from utils.metrics import get_model_complexity, measure_latency, evaluate_model
from utils.visualizer import (
    plot_training_curves, plot_model_comparison,
    plot_pareto_front, plot_ablation_results
)
from utils.logger import ExperimentLogger


# ============================================================================
# Utility Functions
# ============================================================================

def setup_experiment_dir(experiment_type: str, name: str = None) -> Path:
    """Create and return experiment directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if name:
        exp_name = f"{experiment_type}_{name}_{timestamp}"
    else:
        exp_name = f"{experiment_type}_{timestamp}"
    
    exp_dir = PROJECT_ROOT / 'experiments' / 'results' / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)):
    """Print model summary."""
    complexity = get_model_complexity(model, input_size)
    
    print("\n" + "="*50)
    print("Model Summary")
    print("="*50)
    print(f"Parameters: {complexity['params_m']:.2f}M ({complexity['params']:,})")
    print(f"FLOPs: {complexity['flops_m']:.2f}M ({complexity['flops']:,})")
    print(f"Memory: {complexity.get('memory_mb', 0):.2f}MB")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Training Command
# ============================================================================

def cmd_train(args):
    """Run baseline training."""
    print("\n" + "="*60)
    print("Starting Baseline Training")
    print("="*60)
    
    set_seed(args.seed)
    
    # Get config and override with args
    config = get_cifar10_config()
    
    config.train.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.train.learning_rate = args.learning_rate
    config.model.attention_type = args.attention
    config.model.use_attention = args.attention != 'none'
    config.train.experiment_name = f'{args.model}_{args.dataset}_baseline'
    config.train.use_amp = args.amp
    
    if args.dataset == 'cifar100':
        from config import DatasetType
        config.data.dataset = DatasetType.CIFAR100
        config.model.num_classes = 100
    
    # Import and run
    from experiments.baseline import train_baseline
    results = train_baseline(config, model_name=args.model)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Val Acc: {results['best_val_acc']:.2f}%")
    print(f"Final Test Acc: {results['final_test_acc']:.2f}%")
    print("="*60)


# ============================================================================
# Pruning Command
# ============================================================================

def cmd_prune(args):
    """Run pruning experiment."""
    print("\n" + "="*60)
    print("Starting Pruning Experiment")
    print("="*60)
    
    set_seed(args.seed)
    
    sys.argv = [
        'pruning.py',
        '--model', args.model,
        '--attention', args.attention,
        '--dataset', args.dataset,
        '--pruning_type', args.pruning_type,
        '--target_sparsity', str(args.sparsity),
        '--pretrained_path', args.pretrained_path,
        '--save_dir', args.save_dir,
    ]
    
    if args.iterative:
        sys.argv.extend([
            '--iterative',
            '--num_iterations', str(args.num_iterations),
            '--epochs_per_iteration', str(args.epochs_per_iteration)
        ])
    else:
        sys.argv.extend(['--finetune_epochs', str(args.finetune_epochs)])
    
    if args.sensitivity:
        sys.argv.append('--sensitivity_analysis')
    
    from experiments import pruning
    prune_args = pruning.parse_args()
    pruning.run_pruning_experiment(prune_args)


# ============================================================================
# Distillation Command
# ============================================================================

def cmd_distill(args):
    """Run knowledge distillation."""
    print("\n" + "="*60)
    print("Starting Knowledge Distillation")
    print("="*60)
    
    set_seed(args.seed)
    
    sys.argv = [
        'distillation.py',
        '--teacher', args.teacher,
        '--teacher_path', args.teacher_path,
        '--student', args.student,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--temperature', str(args.temperature),
        '--alpha', str(args.alpha),
        '--save_dir', args.save_dir,
    ]
    
    if args.feature_kd:
        sys.argv.extend(['--feature_weight', '0.1'])
    if args.attention_kd:
        sys.argv.extend(['--attention_weight', '0.1'])
    if args.rkd:
        sys.argv.extend(['--rkd_weight', '0.1'])
    
    if args.self_distill:
        sys.argv.extend([
            '--self_distill',
            '--generations', str(args.generations)
        ])
    
    from experiments import distillation
    distill_args = distillation.parse_args()
    
    if distill_args.self_distill:
        # Handle self-distillation separately
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = get_cifar10_config()
        train_loader, val_loader, _, num_classes = get_loaders_from_config(config.data)
        
        model = create_model(distill_args.student, num_classes=num_classes)
        
        save_dir = Path(distill_args.save_dir) / f"self_distill_{distill_args.student}"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger(str(save_dir), "self_distillation")
        
        model, history = distillation.self_distillation(
            model, train_loader, val_loader, device,
            num_generations=distill_args.generations,
            epochs_per_generation=distill_args.epochs // distill_args.generations,
            temperature=distill_args.temperature,
            logger=logger
        )
        
        torch.save(model.state_dict(), save_dir / 'self_distilled_model.pth')
        logger.close()
    else:
        distillation.run_distillation_experiment(distill_args)


# ============================================================================
# Quantization Command
# ============================================================================

def cmd_quantize(args):
    """Run quantization experiment."""
    print("\n" + "="*60)
    print("Starting Quantization Experiment")
    print("="*60)
    
    sys.argv = [
        'quantization.py',
        '--model', args.model,
        '--attention', args.attention,
        '--dataset', args.dataset,
        '--quant_type', args.quant_type,
        '--backend', args.backend,
        '--pretrained_path', args.pretrained_path,
        '--save_dir', args.save_dir,
    ]
    
    if args.quant_type in ['qat', 'fx_qat']:
        sys.argv.extend([
            '--qat_epochs', str(args.qat_epochs),
            '--learning_rate', str(args.learning_rate)
        ])
    
    if args.compare:
        sys.argv.append('--compare_methods')
    
    if args.analyze:
        sys.argv.append('--analyze_error')
    
    from experiments import quantization
    quant_args = quantization.parse_args()
    
    if quant_args.compare_methods:
        quantization.compare_quantization_methods(quant_args)
    else:
        quantization.run_quantization_experiment(quant_args)


# ============================================================================
# Benchmark Command
# ============================================================================

def cmd_benchmark(args):
    """Run comprehensive benchmark."""
    print("\n" + "="*60)
    print("Running Model Benchmark")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get config and data
    config = get_cifar10_config()
    if args.dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    train_loader, val_loader, test_loader = get_loaders_from_config(config.data)
    dataset_name = config.data.dataset.value if hasattr(config.data.dataset, 'value') else str(config.data.dataset)
    num_classes = 100 if dataset_name == 'cifar100' else 10
    
    # Models to benchmark
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = [
            'resnet18', 'resnet34', 'resnet50',
            'ghost_resnet18', 'ghost_resnet_small', 'ghost_resnet_tiny'
        ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\nBenchmarking {model_name}...")
        model = None
        
        try:
            # Only pass attention for ghost models (using attention_type parameter)
            if 'ghost' in model_name.lower():
                model = create_model(model_name, num_classes=num_classes, 
                                   attention_type=args.attention if args.attention != 'none' else 'coordinate')
            else:
                model = create_model(model_name, num_classes=num_classes)
            model = model.to(device)
            
            # Complexity
            complexity = get_model_complexity(model)
            
            # Latency
            input_size = (1, 3, 32, 32) if 'cifar' in args.dataset else (1, 3, 224, 224)
            latency = measure_latency(model, input_size, device=str(device))
            
            # Accuracy (if weights available)
            weight_path = f"weights/{model_name}_best.pth"
            if os.path.exists(weight_path):
                model = load_model_checkpoint(model, weight_path, str(device))
                metrics = evaluate_model(model, test_loader, device)
                accuracy = metrics['accuracy']
            else:
                accuracy = None
            
            results[model_name] = {
                'params_m': complexity['params_m'],
                'flops_m': complexity['flops_m'],
                'latency_ms': latency['mean_ms'],
                'latency_std': latency['std_ms'],
                'accuracy': accuracy
            }
            
            print(f"  Params: {complexity['params_m']:.2f}M")
            print(f"  FLOPs: {complexity['flops_m']:.2f}M")
            print(f"  Latency: {latency['mean_ms']:.2f}ms")
            if accuracy:
                print(f"  Accuracy: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {'error': str(e)}
        
        if model is not None:
            del model
        torch.cuda.empty_cache()
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("Benchmark Summary")
    print("="*80)
    print(f"{'Model':<25} {'Params':<10} {'FLOPs':<10} {'Latency':<12} {'Accuracy':<10}")
    print("-"*80)
    
    for model_name, r in results.items():
        if 'error' not in r:
            acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] else "N/A"
            print(f"{model_name:<25} {r['params_m']:<10.2f}M {r['flops_m']:<10.2f}M "
                  f"{r['latency_ms']:<12.2f}ms {acc_str:<10}")
    
    # Generate comparison plots
    if args.plot:
        # Filter valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v and v.get('accuracy')}
        
        if valid_results:
            # Pareto front
            params = [v['params_m'] for v in valid_results.values()]
            accuracies = [v['accuracy'] for v in valid_results.values()]
            names = list(valid_results.keys())
            
            plot_pareto_front(
                params, accuracies, names,
                xlabel='Parameters (M)', ylabel='Accuracy (%)',
                title='Model Efficiency Pareto Front',
                save_path=str(save_dir / 'pareto_front.png')
            )
            
            # Model comparison
            plot_model_comparison(
                valid_results,
                save_path=str(save_dir / 'model_comparison.png')
            )
    
    print(f"\nResults saved to {save_dir}")


# ============================================================================
# Ablation Study Command
# ============================================================================

def cmd_ablation(args):
    """Run ablation study."""
    print("\n" + "="*60)
    print(f"Running Ablation Study: {args.study}")
    print("="*60)
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get ablation configs
    ablation_configs = get_ablation_configs()
    
    if args.study not in ablation_configs:
        print(f"Error: Unknown ablation study '{args.study}'")
        print(f"Available studies: {list(ablation_configs.keys())}")
        return
    
    configs = ablation_configs[args.study]
    
    # Setup save directory
    save_dir = Path(args.save_dir) / f"ablation_{args.study}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(str(save_dir), f"ablation_{args.study}")
    
    results = {}
    
    for config_name, config in configs.items():
        logger.log(f"\n{'='*50}")
        logger.log(f"Running config: {config_name}")
        logger.log(f"{'='*50}")
        
        try:
            # Create model based on config
            model = create_model(
                config.model.model_type.value if hasattr(config.model.model_type, 'value') 
                else config.model.model_type,
                num_classes=config.data.num_classes,
                attention=config.model.attention if hasattr(config.model, 'attention') else 'none',
                width_multiplier=config.model.width_multiplier if hasattr(config.model, 'width_multiplier') else 1.0,
                drop_path_rate=config.model.drop_path_rate if hasattr(config.model, 'drop_path_rate') else 0.0
            )
            
            # Quick training for ablation (fewer epochs)
            train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
            
            # Simple training loop for ablation
            from experiments.baseline import train_baseline, parse_args as baseline_parse_args
            
            # Configure training
            sys.argv = [
                'baseline.py',
                '--model', config.model.model_type.value if hasattr(config.model.model_type, 'value') 
                          else str(config.model.model_type),
                '--dataset', config.data.dataset,
                '--epochs', str(min(args.epochs, 50)),  # Limit epochs for ablation
                '--batch_size', str(config.data.batch_size),
                '--save_dir', str(save_dir / config_name),
            ]
            
            from experiments import baseline
            train_args = baseline.parse_args()
            
            # Run training
            model_trained, history = baseline.train_baseline(train_args)
            
            # Record results
            best_acc = max(history.get('val_acc', [0]))
            complexity = get_model_complexity(model)
            
            results[config_name] = {
                'accuracy': best_acc,
                'params_m': complexity['params_m'],
                'flops_m': complexity['flops_m'],
                'config': str(config)
            }
            
            logger.log(f"Config {config_name}: Accuracy={best_acc:.2f}%, "
                      f"Params={complexity['params_m']:.2f}M")
            
        except Exception as e:
            logger.log(f"Error in config {config_name}: {e}")
            results[config_name] = {'error': str(e)}
        
        torch.cuda.empty_cache()
    
    # Save results
    with open(save_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    if args.plot:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            names = list(valid_results.keys())
            accuracies = [v['accuracy'] for v in valid_results.values()]
            
            plot_ablation_results(
                names, accuracies,
                title=f'Ablation Study: {args.study}',
                save_path=str(save_dir / 'ablation_plot.png')
            )
    
    # Print summary
    print("\n" + "="*60)
    print(f"Ablation Study Results: {args.study}")
    print("="*60)
    
    for config_name, r in results.items():
        if 'error' not in r:
            print(f"{config_name}: {r['accuracy']:.2f}% ({r['params_m']:.2f}M params)")
        else:
            print(f"{config_name}: ERROR - {r['error']}")
    
    logger.close()
    print(f"\nResults saved to {save_dir}")


# ============================================================================
# Export Command
# ============================================================================

def cmd_export(args):
    """Export model for deployment."""
    print("\n" + "="*60)
    print("Exporting Model for Deployment")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load model
    model = create_model(args.model, num_classes=args.num_classes,
                        attention=args.attention)
    
    if args.weights and os.path.exists(args.weights):
        model = load_model_checkpoint(model, args.weights, str(device))
    
    model.eval()
    
    # Setup save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input size
    if args.input_size:
        input_size = tuple(map(int, args.input_size.split(',')))
    else:
        input_size = (1, 3, 32, 32) if 'cifar' in args.dataset else (1, 3, 224, 224)
    
    example_input = torch.randn(*input_size).to(device)
    model = model.to(device)
    
    if args.format in ['onnx', 'all']:
        # Export to ONNX
        print("\nExporting to ONNX...")
        onnx_path = save_dir / f'{args.model}.onnx'
        
        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"  Saved to {onnx_path}")
        
        # Verify ONNX model
        if args.verify:
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print("  ONNX model verified successfully")
            except Exception as e:
                print(f"  ONNX verification failed: {e}")
    
    if args.format in ['torchscript', 'all']:
        # Export to TorchScript
        print("\nExporting to TorchScript...")
        
        # Traced model
        traced_path = save_dir / f'{args.model}_traced.pt'
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(traced_path))
        print(f"  Traced model saved to {traced_path}")
        
        # Scripted model (more flexible)
        try:
            scripted_path = save_dir / f'{args.model}_scripted.pt'
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(scripted_path))
            print(f"  Scripted model saved to {scripted_path}")
        except Exception as e:
            print(f"  Scripted export failed: {e}")
    
    if args.format in ['mobile', 'all']:
        # Export for mobile
        print("\nExporting for Mobile...")
        
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            
            traced_model = torch.jit.trace(model.cpu(), example_input.cpu())
            optimized_model = optimize_for_mobile(traced_model)
            
            mobile_path = save_dir / f'{args.model}_mobile.pt'
            optimized_model._save_for_lite_interpreter(str(mobile_path))
            print(f"  Mobile model saved to {mobile_path}")
        except Exception as e:
            print(f"  Mobile export failed: {e}")
    
    print(f"\nExport complete. Files saved to {save_dir}")


# ============================================================================
# Full Pipeline Command
# ============================================================================

def cmd_pipeline(args):
    """Run full optimization pipeline."""
    print("\n" + "="*60)
    print("Running Full Optimization Pipeline")
    print("="*60)
    
    set_seed(args.seed)
    
    pipeline_dir = setup_experiment_dir('pipeline', args.name)
    logger = ExperimentLogger(str(pipeline_dir), 'pipeline')
    
    results = {}
    
    # Step 1: Baseline Training
    if 'train' in args.steps or 'all' in args.steps:
        logger.log("\n" + "="*40)
        logger.log("Step 1: Baseline Training")
        logger.log("="*40)
        
        train_args = argparse.Namespace(
            model=args.model,
            attention=args.attention,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=str(pipeline_dir / 'baseline'),
            amp=True,
            mixup=True,
            seed=args.seed
        )
        
        try:
            cmd_train(train_args)
            results['baseline'] = {'status': 'success'}
        except Exception as e:
            logger.log(f"Baseline training failed: {e}")
            results['baseline'] = {'status': 'failed', 'error': str(e)}
    
    # Step 2: Pruning
    if 'prune' in args.steps or 'all' in args.steps:
        logger.log("\n" + "="*40)
        logger.log("Step 2: Pruning")
        logger.log("="*40)
        
        prune_args = argparse.Namespace(
            model=args.model,
            attention=args.attention,
            dataset=args.dataset,
            pruning_type='structured',
            sparsity=0.5,
            pretrained_path=str(pipeline_dir / 'baseline' / 'best_model.pth'),
            save_dir=str(pipeline_dir / 'pruning'),
            iterative=True,
            num_iterations=5,
            epochs_per_iteration=10,
            finetune_epochs=20,
            sensitivity=False,
            seed=args.seed
        )
        
        try:
            cmd_prune(prune_args)
            results['pruning'] = {'status': 'success'}
        except Exception as e:
            logger.log(f"Pruning failed: {e}")
            results['pruning'] = {'status': 'failed', 'error': str(e)}
    
    # Step 3: Quantization
    if 'quantize' in args.steps or 'all' in args.steps:
        logger.log("\n" + "="*40)
        logger.log("Step 3: Quantization")
        logger.log("="*40)
        
        quant_args = argparse.Namespace(
            model=args.model,
            attention=args.attention,
            dataset=args.dataset,
            quant_type='qat',
            backend='fbgemm',
            pretrained_path=str(pipeline_dir / 'pruning' / 'pruned_model.pth'),
            save_dir=str(pipeline_dir / 'quantization'),
            qat_epochs=10,
            learning_rate=0.0001,
            compare=False,
            analyze=True
        )
        
        try:
            cmd_quantize(quant_args)
            results['quantization'] = {'status': 'success'}
        except Exception as e:
            logger.log(f"Quantization failed: {e}")
            results['quantization'] = {'status': 'failed', 'error': str(e)}
    
    # Step 4: Export
    if 'export' in args.steps or 'all' in args.steps:
        logger.log("\n" + "="*40)
        logger.log("Step 4: Export")
        logger.log("="*40)
        
        export_args = argparse.Namespace(
            model=args.model,
            attention=args.attention,
            dataset=args.dataset,
            num_classes=10 if args.dataset == 'cifar10' else 100,
            weights=str(pipeline_dir / 'quantization' / 'quantized_model.pth'),
            format='all',
            save_dir=str(pipeline_dir / 'export'),
            input_size=None,
            verify=True
        )
        
        try:
            cmd_export(export_args)
            results['export'] = {'status': 'success'}
        except Exception as e:
            logger.log(f"Export failed: {e}")
            results['export'] = {'status': 'failed', 'error': str(e)}
    
    # Save pipeline results
    with open(pipeline_dir / 'pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.log("\n" + "="*40)
    logger.log("Pipeline Complete")
    logger.log("="*40)
    
    for step, result in results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        logger.log(f"  {status} {step}: {result['status']}")
    
    logger.close()
    print(f"\nPipeline results saved to {pipeline_dir}")


# ============================================================================
# Main Parser
# ============================================================================

def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='ResNet Optimization Research CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --model resnet18 --dataset cifar10 --epochs 200
  %(prog)s prune --model resnet18 --sparsity 0.5 --iterative
  %(prog)s distill --teacher resnet50 --student ghost_resnet18
  %(prog)s quantize --model resnet18 --type qat
  %(prog)s benchmark --models resnet18 ghost_resnet18 --plot
  %(prog)s ablation --study attention --epochs 50
  %(prog)s export --model resnet18 --format onnx
  %(prog)s pipeline --model resnet18 --steps all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--model', type=str, default='resnet18',
                       help='Model architecture')
    common.add_argument('--attention', type=str, default='none',
                       choices=['none', 'se', 'cbam', 'eca', 'coord', 'triplet'],
                       help='Attention mechanism')
    common.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset')
    common.add_argument('--save_dir', type=str, default='weights',
                       help='Directory to save results')
    common.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[common],
                                         help='Train baseline model')
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--batch_size', type=int, default=128)
    train_parser.add_argument('--learning_rate', type=float, default=0.1)
    train_parser.add_argument('--amp', action='store_true', default=True)
    train_parser.add_argument('--mixup', action='store_true', default=True)
    
    # Prune command
    prune_parser = subparsers.add_parser('prune', parents=[common],
                                         help='Run pruning experiment')
    prune_parser.add_argument('--pruning_type', type=str, default='structured',
                             choices=['structured', 'unstructured', 'random'])
    prune_parser.add_argument('--sparsity', type=float, default=0.5,
                             help='Target sparsity (0-1)')
    prune_parser.add_argument('--pretrained_path', type=str, default='weights/baseline.pth')
    prune_parser.add_argument('--iterative', action='store_true')
    prune_parser.add_argument('--num_iterations', type=int, default=10)
    prune_parser.add_argument('--epochs_per_iteration', type=int, default=5)
    prune_parser.add_argument('--finetune_epochs', type=int, default=20)
    prune_parser.add_argument('--sensitivity', action='store_true')
    
    # Distill command
    distill_parser = subparsers.add_parser('distill', parents=[common],
                                           help='Run knowledge distillation')
    distill_parser.add_argument('--teacher', type=str, default='resnet50')
    distill_parser.add_argument('--teacher_path', type=str, default='weights/teacher.pth')
    distill_parser.add_argument('--student', type=str, default='ghost_resnet18')
    distill_parser.add_argument('--epochs', type=int, default=200)
    distill_parser.add_argument('--temperature', type=float, default=4.0)
    distill_parser.add_argument('--alpha', type=float, default=0.9)
    distill_parser.add_argument('--feature_kd', action='store_true')
    distill_parser.add_argument('--attention_kd', action='store_true')
    distill_parser.add_argument('--rkd', action='store_true')
    distill_parser.add_argument('--self_distill', action='store_true')
    distill_parser.add_argument('--generations', type=int, default=3)
    
    # Quantize command
    quant_parser = subparsers.add_parser('quantize', parents=[common],
                                         help='Run quantization experiment')
    quant_parser.add_argument('--quant_type', type=str, default='ptq',
                             choices=['ptq', 'dynamic', 'qat', 'fx_ptq', 'fx_qat', 'mixed'])
    quant_parser.add_argument('--backend', type=str, default='fbgemm',
                             choices=['fbgemm', 'qnnpack'])
    quant_parser.add_argument('--pretrained_path', type=str, default='weights/baseline.pth')
    quant_parser.add_argument('--qat_epochs', type=int, default=10)
    quant_parser.add_argument('--learning_rate', type=float, default=0.0001)
    quant_parser.add_argument('--compare', action='store_true')
    quant_parser.add_argument('--analyze', action='store_true')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', parents=[common],
                                         help='Run model benchmark')
    bench_parser.add_argument('--models', type=str, nargs='+', default=None,
                             help='Models to benchmark')
    bench_parser.add_argument('--plot', action='store_true')
    
    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', parents=[common],
                                            help='Run ablation study')
    ablation_parser.add_argument('--study', type=str, required=True,
                                choices=['attention', 'width', 'depth', 'augmentation', 'lr'],
                                help='Ablation study type')
    ablation_parser.add_argument('--epochs', type=int, default=50)
    ablation_parser.add_argument('--plot', action='store_true')
    
    # Export command
    export_parser = subparsers.add_parser('export', parents=[common],
                                          help='Export model for deployment')
    export_parser.add_argument('--weights', type=str, default=None,
                              help='Path to model weights')
    export_parser.add_argument('--format', type=str, default='all',
                              choices=['onnx', 'torchscript', 'mobile', 'all'])
    export_parser.add_argument('--num_classes', type=int, default=10)
    export_parser.add_argument('--input_size', type=str, default=None,
                              help='Input size (e.g., "1,3,32,32")')
    export_parser.add_argument('--verify', action='store_true')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', parents=[common],
                                            help='Run full optimization pipeline')
    pipeline_parser.add_argument('--name', type=str, default=None,
                                help='Pipeline name')
    pipeline_parser.add_argument('--steps', type=str, nargs='+', 
                                default=['all'],
                                choices=['train', 'prune', 'quantize', 'export', 'all'])
    pipeline_parser.add_argument('--epochs', type=int, default=200)
    pipeline_parser.add_argument('--batch_size', type=int, default=128)
    pipeline_parser.add_argument('--learning_rate', type=float, default=0.1)
    
    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Print header
    print("\n" + "="*60)
    print("ResNet Optimization Research Framework")
    print("="*60)
    print(f"Command: {args.command}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    # Dispatch to appropriate command
    commands = {
        'train': cmd_train,
        'prune': cmd_prune,
        'distill': cmd_distill,
        'quantize': cmd_quantize,
        'benchmark': cmd_benchmark,
        'ablation': cmd_ablation,
        'export': cmd_export,
        'pipeline': cmd_pipeline
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()