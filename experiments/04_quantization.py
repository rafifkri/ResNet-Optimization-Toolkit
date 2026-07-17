import os
import sys
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from torch.quantization import (
    get_default_qconfig,
    get_default_qat_qconfig,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic,
    QuantStub,
    DeQuantStub
)
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_cifar10_config, QuantizationType
from models import create_model
from utils.dataloader import get_loaders_from_config, get_calibration_loader
from utils.metrics import get_model_complexity, measure_latency, evaluate_model
from utils.logger import ExperimentLogger, AverageMeter
from utils.checkpoint import CheckpointManager
from utils.augmentations import LabelSmoothingCrossEntropy


# ============================================================================
# Quantization-Ready Model Wrapper
# ============================================================================

class QuantizableModel(nn.Module):
    """
    Wrapper to make any model quantization-ready.
    Adds QuantStub and DeQuantStub for proper quantization.
    """
    
    def __init__(self, model: nn.Module, num_classes: int = 10):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv+BN+ReLU layers for better quantization."""
        # This is a generic fusion - specific models may need custom fusion
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Sequential):
                # Try to fuse conv-bn-relu patterns
                modules_to_fuse = []
                for idx, (name, m) in enumerate(module.named_children()):
                    if isinstance(m, nn.Conv2d):
                        fuse_list = [name]
                        children = list(module.named_children())
                        if idx + 1 < len(children):
                            next_name, next_m = children[idx + 1]
                            if isinstance(next_m, nn.BatchNorm2d):
                                fuse_list.append(next_name)
                                if idx + 2 < len(children):
                                    next_next_name, next_next_m = children[idx + 2]
                                    if isinstance(next_next_m, nn.ReLU):
                                        fuse_list.append(next_next_name)
                        if len(fuse_list) > 1:
                            modules_to_fuse.append(fuse_list)
                
                for fuse_list in modules_to_fuse:
                    try:
                        torch.quantization.fuse_modules(module, fuse_list, inplace=True)
                    except Exception:
                        pass  # Skip if fusion fails


def create_quantizable_model(model_name: str, num_classes: int = 10, 
                            attention: str = 'none', **kwargs) -> QuantizableModel:
    """Create a quantization-ready model."""
    # Remove attention from kwargs if it exists to avoid duplicate argument
    kwargs.pop('attention', None)
    kwargs.pop('attention_type', None)
    # Only pass attention to ghost models that support it
    if 'ghost' in model_name.lower():
        base_model = create_model(model_name, num_classes=num_classes, 
                                 attention_type=attention, **kwargs)
    else:
        base_model = create_model(model_name, num_classes=num_classes, **kwargs)
    return QuantizableModel(base_model, num_classes)


# ============================================================================
# Post-Training Quantization (PTQ)
# ============================================================================

def post_training_static_quantization(
    model: nn.Module,
    calibration_loader,
    backend: str = 'fbgemm',
    num_calibration_batches: int = 100
) -> nn.Module:
    """
    Apply Post-Training Static Quantization (PTQ).
    
    This quantizes weights and activations to INT8 using calibration data.
    Best for inference speed on CPU.
    
    Args:
        model: Model to quantize (should be QuantizableModel)
        calibration_loader: DataLoader for calibration
        backend: 'fbgemm' for x86, 'qnnpack' for ARM
        num_calibration_batches: Number of batches for calibration
    
    Returns:
        Quantized INT8 model
    """
    model.eval()
    model.cpu()  # Quantization currently only works on CPU
    
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Set quantization config
    model.qconfig = get_default_qconfig(backend)
    
    # Fuse modules (Conv+BN+ReLU)
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Prepare for calibration
    model_prepared = prepare(model, inplace=False)
    
    # Calibrate with representative data
    print(f"Calibrating with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(calibration_loader, 
                                              total=num_calibration_batches,
                                              desc="Calibration")):
            if i >= num_calibration_batches:
                break
            inputs = inputs.cpu()
            model_prepared(inputs)
    
    # Convert to quantized model
    model_quantized = convert(model_prepared, inplace=False)
    
    return model_quantized


def post_training_dynamic_quantization(
    model: nn.Module,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Apply Post-Training Dynamic Quantization.
    
    Only quantizes weights (activations computed in float, quantized on-the-fly).
    Simpler but less efficient than static quantization.
    Good for models with variable input sizes or when calibration data is limited.
    
    Args:
        model: Model to quantize
        backend: Quantization backend
    
    Returns:
        Dynamically quantized model
    """
    model.eval()
    model.cpu()
    
    torch.backends.quantized.engine = backend
    
    # Apply dynamic quantization to Linear and LSTM layers
    model_quantized = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8
    )
    
    return model_quantized


# ============================================================================
# Quantization-Aware Training (QAT)
# ============================================================================

def quantization_aware_training(
    model: nn.Module,
    train_loader,
    val_loader,
    num_classes: int,
    device: str = 'cuda',
    epochs: int = 10,
    learning_rate: float = 0.0001,
    backend: str = 'fbgemm',
    logger: ExperimentLogger = None
) -> Tuple[nn.Module, Dict]:
    """
    Quantization-Aware Training (QAT).
    
    Fine-tunes the model with simulated quantization during training.
    Achieves better accuracy than PTQ, especially for aggressive quantization.
    
    Args:
        model: Model to train (should be QuantizableModel)
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        device: Training device
        epochs: Number of QAT epochs
        learning_rate: Learning rate (should be small)
        backend: Quantization backend
        logger: Experiment logger
    
    Returns:
        Quantized model and training history
    """
    # Setup
    torch.backends.quantized.engine = backend
    
    # Clone model and move to CPU for preparation
    model_qat = copy.deepcopy(model)
    model_qat.cpu()
    model_qat.train()
    
    # Set QAT config
    model_qat.qconfig = get_default_qat_qconfig(backend)
    
    # Fuse modules
    if hasattr(model_qat, 'fuse_model'):
        model_qat.fuse_model()
    
    # Prepare for QAT
    model_qat = prepare_qat(model_qat, inplace=False)
    
    # Move to device for training
    model_qat = model_qat.to(device)
    
    # Setup training
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model_qat.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model_qat.train()
        train_loss = AverageMeter('Loss')
        train_acc = AverageMeter('Acc')
        
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model_qat(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            acc = (outputs.argmax(1) == targets).float().mean().item() * 100
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc, inputs.size(0))
            
            pbar.set_postfix({'loss': f'{train_loss.avg:.3f}', 'acc': f'{train_acc.avg:.1f}'})
        
        scheduler.step()
        
        # Validate
        val_acc = evaluate_quantized(model_qat, val_loader, device)
        
        if logger:
            logger.info(f"QAT Epoch {epoch}: Train Loss={train_loss.avg:.3f}, "
                      f"Train Acc={train_acc.avg:.1f}%, Val Acc={val_acc:.2f}%")
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model_qat.state_dict())
    
    # Load best state
    if best_state is not None:
        model_qat.load_state_dict(best_state)
    
    # Convert to quantized model (must be on CPU)
    model_qat.cpu()
    model_qat.eval()
    model_quantized = convert(model_qat, inplace=False)
    
    return model_quantized, history


def evaluate_quantized(model: nn.Module, dataloader, device: str = 'cpu') -> float:
    """Evaluate a (possibly quantized) model."""
    model.eval()
    
    # Determine if model needs to be on CPU (quantized models)
    is_quantized = any(
        isinstance(m, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear))
        for m in model.modules()
    )
    
    if is_quantized:
        model = model.cpu()
        device = 'cpu'
    else:
        model = model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device if not is_quantized else 'cpu')
            
            outputs = model(inputs if not is_quantized else inputs.cpu())
            if is_quantized:
                outputs = outputs.to(targets.device)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


# ============================================================================
# FX Graph Mode Quantization (Recommended)
# ============================================================================

def fx_graph_mode_quantization(
    model: nn.Module,
    calibration_loader,
    backend: str = 'fbgemm',
    num_calibration_batches: int = 100,
    qconfig_mapping: str = 'default'
) -> nn.Module:
    """
    FX Graph Mode Quantization.
    
    More flexible and automatic than eager mode quantization.
    Handles complex model architectures better.
    
    Args:
        model: Model to quantize
        calibration_loader: Calibration data loader
        backend: Quantization backend
        num_calibration_batches: Number of calibration batches
        qconfig_mapping: 'default' or 'per_channel'
    
    Returns:
        Quantized model
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    
    model.eval()
    model.cpu()
    
    torch.backends.quantized.engine = backend
    
    # Get qconfig mapping
    qconfig_mapping = get_default_qconfig_mapping(backend)
    
    # Get example input for tracing
    example_input = next(iter(calibration_loader))[0][:1].cpu()
    
    # Prepare model
    model_prepared = prepare_fx(model, qconfig_mapping, example_input)
    
    # Calibrate
    print(f"Calibrating FX model with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(calibration_loader,
                                              total=num_calibration_batches,
                                              desc="FX Calibration")):
            if i >= num_calibration_batches:
                break
            model_prepared(inputs.cpu())
    
    # Convert
    model_quantized = convert_fx(model_prepared)
    
    return model_quantized


def fx_qat(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    learning_rate: float = 0.0001,
    backend: str = 'fbgemm',
    logger: ExperimentLogger = None
) -> Tuple[nn.Module, Dict]:
    """
    FX Graph Mode Quantization-Aware Training.
    
    Combines FX graph mode with QAT for best results.
    """
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    
    model.train()
    model.cpu()
    
    torch.backends.quantized.engine = backend
    
    # Get QAT qconfig mapping
    qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    
    # Get example input
    example_input = next(iter(train_loader))[0][:1].cpu()
    
    # Prepare for QAT
    model_prepared = prepare_qat_fx(model, qconfig_mapping, example_input)
    
    # Move to GPU for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_prepared = model_prepared.to(device)
    
    # Training setup
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model_prepared.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'epoch': [], 'train_loss': [], 'val_acc': []}
    best_acc = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model_prepared.train()
        train_loss = AverageMeter('Loss')
        
        pbar = tqdm(train_loader, desc=f"FX-QAT Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), inputs.size(0))
            pbar.set_postfix({'loss': f'{train_loss.avg:.3f}'})
        
        scheduler.step()
        
        # Validate (convert temporarily for accurate validation)
        model_prepared.cpu()
        model_prepared.eval()
        
        # We'll validate the QAT model directly (not converted)
        val_acc = evaluate_quantized(model_prepared.to(device), val_loader, str(device))
        model_prepared = model_prepared.to(device)
        
        if logger:
            logger.info(f"FX-QAT Epoch {epoch}: Train Loss={train_loss.avg:.3f}, Val Acc={val_acc:.2f}%")
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.avg)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model_prepared.state_dict())
    
    # Load best and convert
    if best_state:
        model_prepared.load_state_dict(best_state)
    
    model_prepared.cpu()
    model_prepared.eval()
    model_quantized = convert_fx(model_prepared)
    
    return model_quantized, history


# ============================================================================
# Mixed Precision Quantization
# ============================================================================

def mixed_precision_quantization(
    model: nn.Module,
    calibration_loader,
    sensitive_layers: List[str] = None,
    backend: str = 'fbgemm',
    num_calibration_batches: int = 100
) -> nn.Module:
    """
    Mixed Precision Quantization.
    
    Quantizes most layers to INT8 but keeps sensitive layers in FP32.
    Useful when some layers are sensitive to quantization.
    
    Args:
        model: Model to quantize
        calibration_loader: Calibration data loader
        sensitive_layers: List of layer names to keep in FP32
        backend: Quantization backend
        num_calibration_batches: Calibration batches
    
    Returns:
        Mixed precision quantized model
    """
    if sensitive_layers is None:
        # Default: keep first and last layers in FP32
        sensitive_layers = []
    
    model.eval()
    model.cpu()
    
    torch.backends.quantized.engine = backend
    
    # Set per-layer qconfig
    for name, module in model.named_modules():
        if any(sens in name for sens in sensitive_layers):
            # Keep in FP32
            module.qconfig = None
        else:
            module.qconfig = get_default_qconfig(backend)
    
    # Fuse if possible
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Prepare and calibrate
    model_prepared = prepare(model, inplace=False)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(calibration_loader,
                                              total=num_calibration_batches,
                                              desc="Mixed-Precision Calibration")):
            if i >= num_calibration_batches:
                break
            model_prepared(inputs.cpu())
    
    # Convert
    model_quantized = convert(model_prepared, inplace=False)
    
    return model_quantized


# ============================================================================
# Quantization Analysis
# ============================================================================

def analyze_quantization_error(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    dataloader,
    num_batches: int = 50
) -> Dict:
    """
    Analyze quantization error between FP32 and quantized models.
    
    Computes per-layer error statistics.
    """
    fp32_model.eval()
    quantized_model.eval()
    
    fp32_model.cpu()
    quantized_model.cpu()
    
    errors = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            inputs = inputs.cpu()
            
            fp32_out = fp32_model(inputs)
            quant_out = quantized_model(inputs)
            
            # Compute relative error
            error = (fp32_out - quant_out).abs()
            rel_error = error / (fp32_out.abs() + 1e-8)
            
            errors.append({
                'mae': error.mean().item(),
                'max_error': error.max().item(),
                'mean_rel_error': rel_error.mean().item()
            })
    
    # Aggregate
    analysis = {
        'mean_absolute_error': np.mean([e['mae'] for e in errors]),
        'max_absolute_error': np.max([e['max_error'] for e in errors]),
        'mean_relative_error': np.mean([e['mean_rel_error'] for e in errors])
    }
    
    return analysis


def get_quantized_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    # Save to buffer to measure size
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return size_mb


def compare_models(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader,
    device: str = 'cuda'
) -> Dict:
    """Compare FP32 and quantized models."""
    
    # FP32 evaluation
    fp32_model = fp32_model.to(device)
    fp32_acc = evaluate_quantized(fp32_model, test_loader, device)
    fp32_latency = measure_latency(fp32_model, (1, 3, 32, 32), device=device)
    fp32_size = get_quantized_model_size(fp32_model)
    
    # Quantized evaluation (on CPU)
    quantized_model = quantized_model.cpu()
    quant_acc = evaluate_quantized(quantized_model, test_loader, 'cpu')
    quant_latency = measure_latency(quantized_model, (1, 3, 32, 32), device='cpu')
    quant_size = get_quantized_model_size(quantized_model)
    
    comparison = {
        'fp32': {
            'accuracy': fp32_acc,
            'latency_ms': fp32_latency['mean_ms'],
            'size_mb': fp32_size
        },
        'quantized': {
            'accuracy': quant_acc,
            'latency_ms': quant_latency['mean_ms'],
            'size_mb': quant_size
        },
        'improvement': {
            'accuracy_drop': fp32_acc - quant_acc,
            'speedup': fp32_latency['mean_ms'] / max(quant_latency['mean_ms'], 1e-6),
            'compression': fp32_size / max(quant_size, 1e-6)
        }
    }
    
    return comparison


# ============================================================================
# Main Experiment Function
# ============================================================================

def run_quantization_experiment(args):
    """Run complete quantization experiment."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Quantization backend: {args.backend}")
    
    # Get config
    config = get_cifar10_config()
    if args.dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    # Setup directories
    experiment_name = f"quantization_{args.quant_type}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger - disable TensorBoard due to NumPy 2.x compatibility issues
    logger = ExperimentLogger(experiment_name, str(save_dir), use_tensorboard=False)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_loaders_from_config(config.data)
    dataset_name = config.data.dataset.value if hasattr(config.data.dataset, 'value') else str(config.data.dataset)
    num_classes = 100 if dataset_name == 'cifar100' else 10
    eval_loader = val_loader if val_loader is not None else test_loader
    calibration_loader = get_calibration_loader(
        batch_size=32,
        num_batches=args.num_calibration_batches,
        dataset=dataset_name
    )
    
    # Create model
    logger.info("\n" + "="*50)
    logger.info("Loading Model")
    logger.info("="*50)
    
    if args.quant_type in ['ptq', 'qat', 'dynamic']:
        # Use quantizable wrapper for eager mode
        model = create_quantizable_model(
            args.model,
            num_classes=num_classes,
            attention=args.attention
        )
    else:
        # FX mode can handle regular models
        model = create_model(
            args.model,
            num_classes=num_classes,
            attention_type=args.attention
        )
    
    # Load pretrained weights
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        logger.info(f"Loading weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
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
        
        # Handle wrapper vs direct model
        if hasattr(model, 'model'):
            # QuantizableModel wrapper
            try:
                model.model.load_state_dict(state_dict, strict=False)
            except:
                model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {len(state_dict)} parameters from checkpoint")
    else:
        logger.info("No pretrained weights loaded - using random initialization")
    
    # Initial evaluation
    model_eval = model.model if hasattr(model, 'model') else model
    model_eval = model_eval.to(device)
    fp32_acc = evaluate_quantized(model_eval, eval_loader, str(device))
    fp32_complexity = get_model_complexity(model_eval)
    
    logger.info(f"\nFP32 Model:")
    logger.info(f"  Accuracy: {fp32_acc:.2f}%")
    logger.info(f"  Parameters: {fp32_complexity['params_m']:.2f}M")
    logger.info(f"  Size: {get_quantized_model_size(model_eval):.2f}MB")
    
    # Apply quantization
    logger.info("\n" + "="*50)
    logger.info(f"Applying {args.quant_type.upper()} Quantization")
    logger.info("="*50)
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    if args.quant_type == 'ptq':
        # Post-Training Static Quantization
        quantized_model = post_training_static_quantization(
            model,
            calibration_loader,
            backend=args.backend,
            num_calibration_batches=args.num_calibration_batches
        )
        history = None
        
    elif args.quant_type == 'dynamic':
        # Dynamic Quantization
        quantized_model = post_training_dynamic_quantization(
            model,
            backend=args.backend
        )
        history = None
        
    elif args.quant_type == 'qat':
        # Quantization-Aware Training
        quantized_model, history = quantization_aware_training(
            model,
            train_loader,
            val_loader,
            num_classes,
            device=str(device),
            epochs=args.qat_epochs,
            learning_rate=args.learning_rate,
            backend=args.backend,
            logger=logger
        )
        
    elif args.quant_type == 'fx_ptq':
        # FX Graph Mode PTQ
        base_model = model.model if hasattr(model, 'model') else model
        quantized_model = fx_graph_mode_quantization(
            base_model,
            calibration_loader,
            backend=args.backend,
            num_calibration_batches=args.num_calibration_batches
        )
        history = None
        
    elif args.quant_type == 'fx_qat':
        # FX Graph Mode QAT
        base_model = model.model if hasattr(model, 'model') else model
        quantized_model, history = fx_qat(
            base_model,
            train_loader,
            val_loader,
            epochs=args.qat_epochs,
            learning_rate=args.learning_rate,
            backend=args.backend,
            logger=logger
        )
        
    elif args.quant_type == 'mixed':
        # Mixed Precision Quantization
        quantized_model = mixed_precision_quantization(
            model,
            calibration_loader,
            sensitive_layers=args.sensitive_layers,
            backend=args.backend,
            num_calibration_batches=args.num_calibration_batches
        )
        history = None
    
    else:
        raise ValueError(f"Unknown quantization type: {args.quant_type}")
    
    # Evaluate quantized model
    logger.info("\n" + "="*50)
    logger.info("Quantized Model Evaluation")
    logger.info("="*50)
    
    quantized_model = quantized_model.cpu()
    quant_acc = evaluate_quantized(quantized_model, test_loader, 'cpu')
    quant_size = get_quantized_model_size(quantized_model)
    quant_latency = measure_latency(quantized_model, (1, 3, 32, 32), device='cpu')
    
    logger.info(f"Quantized Model (INT8):")
    logger.info(f"  Test Accuracy: {quant_acc:.2f}%")
    logger.info(f"  Size: {quant_size:.2f}MB")
    logger.info(f"  Latency (CPU): {quant_latency['mean_ms']:.2f} Â± {quant_latency['std_ms']:.2f} ms")
    
    # Calculate improvements
    fp32_size = get_quantized_model_size(model_eval.cpu())
    compression = fp32_size / max(quant_size, 1e-6)
    accuracy_drop = fp32_acc - quant_acc
    
    logger.info(f"\nImprovements:")
    logger.info(f"  Compression ratio: {compression:.2f}x")
    logger.info(f"  Size reduction: {(1 - quant_size/fp32_size)*100:.1f}%")
    logger.info(f"  Accuracy drop: {accuracy_drop:.2f}%")
    
    # Analyze quantization error
    if args.analyze_error:
        logger.info("\nQuantization Error Analysis:")
        error_analysis = analyze_quantization_error(
            model, quantized_model, calibration_loader
        )
        for key, value in error_analysis.items():
            logger.info(f"  {key}: {value:.6f}")
    
    # Save results
    summary = {
        'fp32': {
            'accuracy': fp32_acc,
            'params_m': fp32_complexity['params_m'],
            'size_mb': fp32_size
        },
        'quantized': {
            'accuracy': quant_acc,
            'size_mb': quant_size,
            'latency_ms': quant_latency['mean_ms']
        },
        'improvements': {
            'compression_ratio': compression,
            'accuracy_drop': accuracy_drop,
            'size_reduction_percent': (1 - quant_size/fp32_size) * 100
        },
        'config': {
            'quant_type': args.quant_type,
            'backend': args.backend,
            'model': args.model
        }
    }
    
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if history:
        with open(save_dir / 'qat_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), save_dir / 'quantized_model.pth')
    
    # Save as TorchScript for deployment
    try:
        example_input = torch.randn(1, 3, 32, 32)
        traced_model = torch.jit.trace(quantized_model, example_input)
        traced_model.save(str(save_dir / 'quantized_model_scripted.pt'))
        logger.info(f"\nTorchScript model saved to {save_dir / 'quantized_model_scripted.pt'}")
    except Exception as e:
        logger.info(f"\nWarning: Could not save TorchScript model: {e}")
    
    logger.info(f"\nResults saved to {save_dir}")
    logger.close()
    
    return quantized_model, summary


# ============================================================================
# Comparison Experiment
# ============================================================================

def compare_quantization_methods(args):
    """Compare different quantization methods."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = get_cifar10_config()
    train_loader, val_loader, test_loader, num_classes = get_loaders_from_config(config.data)
    calibration_loader = get_calibration_loader(
        config.data.data_dir,
        config.data.dataset,
        batch_size=32,
        num_samples=1000
    )
    
    results = {}
    methods = ['ptq', 'dynamic', 'qat'] if args.include_qat else ['ptq', 'dynamic']
    
    # Create base model
    base_model = create_model(args.model, num_classes=num_classes)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint)
    
    # FP32 baseline
    base_model = base_model.to(device)
    fp32_acc = evaluate_quantized(base_model, test_loader, str(device))
    fp32_size = get_quantized_model_size(base_model.cpu())
    
    results['fp32'] = {
        'accuracy': fp32_acc,
        'size_mb': fp32_size
    }
    
    print(f"FP32 Baseline: {fp32_acc:.2f}%, {fp32_size:.2f}MB")
    
    for method in methods:
        print(f"\nTesting {method.upper()}...")
        
        # Create fresh quantizable model
        model = create_quantizable_model(args.model, num_classes=num_classes)
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if method == 'ptq':
            quantized = post_training_static_quantization(
                model, calibration_loader, backend=args.backend
            )
        elif method == 'dynamic':
            quantized = post_training_dynamic_quantization(
                model, backend=args.backend
            )
        elif method == 'qat':
            quantized, _ = quantization_aware_training(
                model, train_loader, val_loader, num_classes,
                device=str(device), epochs=5, backend=args.backend
            )
        
        acc = evaluate_quantized(quantized, test_loader, 'cpu')
        size = get_quantized_model_size(quantized)
        
        results[method] = {
            'accuracy': acc,
            'size_mb': size,
            'accuracy_drop': fp32_acc - acc,
            'compression': fp32_size / size
        }
        
        print(f"  Accuracy: {acc:.2f}% (drop: {fp32_acc - acc:.2f}%)")
        print(f"  Size: {size:.2f}MB (compression: {fp32_size/size:.2f}x)")
        
        del quantized
        torch.cuda.empty_cache()
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'quantization_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Quantization Method Comparison Summary")
    print("="*60)
    print(f"{'Method':<15} {'Accuracy':<12} {'Size (MB)':<12} {'Compression':<12}")
    print("-"*60)
    print(f"{'FP32':<15} {results['fp32']['accuracy']:<12.2f} {results['fp32']['size_mb']:<12.2f} {'1.00x':<12}")
    for method in methods:
        r = results[method]
        print(f"{method.upper():<15} {r['accuracy']:<12.2f} {r['size_mb']:<12.2f} {r['compression']:<12.2f}x")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization Experiment')
    
    # Model
    parser.add_argument('--model', type=str, default='ghost_resnet18',
                        choices=['resnet18', 'resnet34', 'ghost_resnet18', 'ghost_resnet_small'],
                        help='Model architecture')
    parser.add_argument('--attention', type=str, default='coordinate',
                        choices=['none', 'se', 'cbam', 'eca', 'coordinate', 'coord'])
    parser.add_argument('--pretrained_path', type=str, default='weights/baseline.pth',
                        help='Path to pretrained model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    
    # Quantization
    parser.add_argument('--quant_type', type=str, default='dynamic',
                        choices=['ptq', 'dynamic', 'qat', 'fx_ptq', 'fx_qat', 'mixed'],
                        help='Quantization type (dynamic recommended for ResNet skip connections)')
    parser.add_argument('--backend', type=str, default='fbgemm',
                        choices=['fbgemm', 'qnnpack'],
                        help='Quantization backend (fbgemm for x86, qnnpack for ARM)')
    
    # Calibration
    parser.add_argument('--num_calibration_samples', type=int, default=1000,
                        help='Number of samples for calibration')
    parser.add_argument('--num_calibration_batches', type=int, default=100,
                        help='Number of batches for calibration')
    
    # QAT
    parser.add_argument('--qat_epochs', type=int, default=10,
                        help='Number of QAT epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for QAT')
    
    # Mixed precision
    parser.add_argument('--sensitive_layers', type=str, nargs='+', default=None,
                        help='Layers to keep in FP32 for mixed precision')
    
    # Analysis
    parser.add_argument('--analyze_error', action='store_true',
                        help='Analyze quantization error')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare different quantization methods')
    parser.add_argument('--include_qat', action='store_true',
                        help='Include QAT in method comparison')
    
    # Save
    parser.add_argument('--save_dir', type=str, default='weights/quantization_experiments')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.compare_methods:
        compare_quantization_methods(args)
    else:
        run_quantization_experiment(args)
