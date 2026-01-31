"""
===================================================================================
Metrics Module - Model Analysis and Evaluation
===================================================================================
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from collections import OrderedDict


# ===================================================================================
# MODEL COMPLEXITY METRICS
# ===================================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_m': total / 1e6,
        'trainable_m': trainable / 1e6,
        'params_m': total / 1e6,  # Alias for compatibility
    }


def get_model_complexity(model: nn.Module, 
                         input_size: Tuple[int, ...] = (1, 3, 32, 32),
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Calculate model complexity (FLOPs, MACs, parameters).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        verbose: Print detailed info
    
    Returns:
        Dictionary with complexity metrics
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    result = count_parameters(model)
    
    # Set default FLOPs values (will be overwritten if thop is available)
    result['flops'] = 0
    result['flops_m'] = 0
    result['flops_g'] = 0
    result['macs'] = 0
    result['macs_m'] = 0
    
    # Try thop
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        result['flops'] = flops
        result['flops_m'] = flops / 1e6
        result['flops_g'] = flops / 1e9
        result['macs'] = flops / 2  # MACs ≈ FLOPs / 2
        result['macs_m'] = flops / 2 / 1e6
        
        if verbose:
            flops_str, params_str = clever_format([flops, params], "%.3f")
            print(f"FLOPs: {flops_str}, Params: {params_str}")
    except ImportError:
        pass
    
    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, dummy_input)
        result['fvcore_flops'] = flop_counter.total()
    except ImportError:
        pass
    
    return result


def measure_latency(model: nn.Module,
                    input_size: Tuple[int, ...] = (1, 3, 32, 32),
                    device: str = 'cpu',
                    iterations: int = 100,
                    warmup: int = 20) -> Dict[str, float]:
    """
    Measure model inference latency.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to benchmark on
        iterations: Number of iterations
        warmup: Warmup iterations
    
    Returns:
        Dictionary with latency metrics
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'throughput': 1000 / np.mean(latencies) * input_size[0],  # samples/sec
    }


def get_model_size(model: nn.Module, 
                   as_mb: bool = True) -> Dict[str, float]:
    """
    Calculate model size in memory.
    
    Args:
        model: PyTorch model
        as_mb: Return size in MB
    
    Returns:
        Dictionary with size metrics
    """
    # Parameter size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Buffer size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    divisor = 1024 * 1024 if as_mb else 1
    unit = 'MB' if as_mb else 'bytes'
    
    return {
        f'param_size_{unit}': param_size / divisor,
        f'buffer_size_{unit}': buffer_size / divisor,
        f'total_size_{unit}': total_size / divisor,
    }


# ===================================================================================
# CLASSIFICATION METRICS
# ===================================================================================

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        topk: Tuple of k values
    
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


class ClassificationMetrics:
    """
    Track classification metrics during training/evaluation.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
        self.correct = 0
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results"""
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            self.all_predictions.extend(preds.cpu().numpy())
            self.all_targets.extend(targets.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())
            
            self.total_samples += targets.size(0)
            self.correct += preds.eq(targets).sum().item()
            
            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics"""
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)
        
        # Basic accuracy
        acc = self.correct / self.total_samples if self.total_samples > 0 else 0
        
        # Per-class metrics
        per_class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        
        # Precision, Recall, F1 for each class
        precision = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + 1e-10)
        recall = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'accuracy': acc * 100,
            'per_class_accuracy': per_class_acc * 100,
            'mean_class_accuracy': np.mean(per_class_acc) * 100,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': np.mean(f1),
            'confusion_matrix': self.confusion_matrix,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probs,
        }


# ===================================================================================
# MODEL EVALUATION
# ===================================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   criterion: Optional[nn.Module] = None,
                   device: torch.device = torch.device('cpu'),
                   num_classes: int = 10) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        data_loader: Test/validation data loader
        criterion: Loss function
        device: Device to evaluate on
        num_classes: Number of classes
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model = model.to(device)
    
    metrics = ClassificationMetrics(num_classes)
    total_loss = 0
    num_batches = 0
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        if criterion is not None:
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        
        metrics.update(outputs, targets)
        num_batches += 1
    
    results = metrics.compute()
    
    if criterion is not None:
        results['loss'] = total_loss / num_batches
    
    return results


# ===================================================================================
# COMPARISON UTILITIES
# ===================================================================================

def compare_models(models: Dict[str, nn.Module],
                   input_size: Tuple[int, ...] = (1, 3, 32, 32),
                   device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        input_size: Input size for benchmarking
        device: Device for latency measurement
    
    Returns:
        Comparison results
    """
    results = {}
    
    for name, model in models.items():
        print(f"Analyzing {name}...")
        
        model_results = {}
        model_results.update(get_model_complexity(model, input_size))
        model_results.update(measure_latency(model, input_size, device))
        model_results.update(get_model_size(model))
        
        results[name] = model_results
    
    return results


def print_comparison_table(comparison: Dict[str, Dict[str, Any]],
                           metrics: List[str] = None):
    """
    Print comparison table.
    
    Args:
        comparison: Output from compare_models
        metrics: List of metrics to show
    """
    if metrics is None:
        metrics = ['total_m', 'flops_m', 'mean_ms', 'total_size_MB']
    
    # Header
    header = ['Model'] + metrics
    print('\t'.join(header))
    print('-' * 80)
    
    # Rows
    for name, results in comparison.items():
        row = [name]
        for m in metrics:
            val = results.get(m, 'N/A')
            if isinstance(val, float):
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        print('\t'.join(row))


# ===================================================================================
# SPARSITY ANALYSIS (for pruning)
# ===================================================================================

def get_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model sparsity (percentage of zero weights).
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with sparsity metrics
    """
    total_params = 0
    zero_params = 0
    
    layer_sparsity = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total = param.numel()
            zeros = (param == 0).sum().item()
            
            total_params += total
            zero_params += zeros
            layer_sparsity[name] = zeros / total * 100
    
    return {
        'global_sparsity': zero_params / total_params * 100 if total_params > 0 else 0,
        'layer_sparsity': layer_sparsity,
        'total_params': total_params,
        'zero_params': zero_params,
    }


# ===================================================================================
# TESTING
# ===================================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from models import resnet18_base, ghost_resnet18
    
    print("Testing metrics module...")
    
    # Create models
    baseline = resnet18_base(num_classes=10)
    ghost = ghost_resnet18(num_classes=10)
    
    # Compare
    comparison = compare_models(
        {'ResNet-18': baseline, 'Ghost-ResNet-18': ghost},
        input_size=(1, 3, 32, 32)
    )
    
    print("\nModel Comparison:")
    print_comparison_table(comparison)
    
    # Test accuracy function
    output = torch.randn(32, 10)
    target = torch.randint(0, 10, (32,))
    top1, top5 = accuracy(output, target, topk=(1, 5))
    print(f"\nRandom accuracy: Top-1={top1:.2f}%, Top-5={top5:.2f}%")