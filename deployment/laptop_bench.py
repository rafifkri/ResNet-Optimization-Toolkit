"""
laptop_bench.py - Comprehensive Benchmarking for Laptop/Desktop Deployment

This script provides thorough benchmarking capabilities:
1. PyTorch Native Inference
2. ONNX Runtime Inference
3. TensorRT Inference (if available)
4. OpenVINO Inference (if available)
5. Multi-threading/CPU utilization analysis
6. Memory profiling
7. Comparative analysis with visualization

Reference:
- ONNX Runtime: https://onnxruntime.ai/
- TensorRT: https://developer.nvidia.com/tensorrt
- OpenVINO: https://docs.openvino.ai/
"""

import os
import sys
import json
import argparse
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model
from utils.metrics import get_model_complexity


# ============================================================================
# PyTorch Benchmarking
# ============================================================================

def benchmark_pytorch(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    num_warmup: int = 50,
    num_iterations: int = 200,
    batch_sizes: List[int] = [1, 8, 16, 32]
) -> Dict:
    """
    Benchmark PyTorch model inference.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to use ('cpu' or 'cuda')
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        batch_sizes: List of batch sizes to test
    
    Returns:
        Benchmark results dictionary
    """
    model.eval()
    model = model.to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        full_input_size = (batch_size,) + input_size
        input_tensor = torch.randn(*full_input_size, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(input_tensor)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        results[f'batch_{batch_size}'] = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': batch_size / (np.mean(latencies) / 1000)
        }
    
    return results


def benchmark_pytorch_amp(
    model: nn.Module,
    input_size: Tuple[int, ...],
    num_warmup: int = 50,
    num_iterations: int = 200,
    batch_size: int = 32
) -> Dict:
    """
    Benchmark PyTorch with Automatic Mixed Precision (AMP).
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available for AMP'}
    
    model.eval()
    model = model.cuda()
    
    full_input_size = (batch_size,) + input_size
    input_tensor = torch.randn(*full_input_size, device='cuda')
    
    # FP32 baseline
    latencies_fp32 = []
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_fp32.append((end - start) * 1000)
    
    # FP16 with AMP
    latencies_fp16 = []
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for _ in range(num_warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_fp16.append((end - start) * 1000)
    
    return {
        'fp32': {
            'mean_ms': np.mean(latencies_fp32),
            'std_ms': np.std(latencies_fp32),
            'throughput': batch_size / (np.mean(latencies_fp32) / 1000)
        },
        'fp16_amp': {
            'mean_ms': np.mean(latencies_fp16),
            'std_ms': np.std(latencies_fp16),
            'throughput': batch_size / (np.mean(latencies_fp16) / 1000)
        },
        'speedup': np.mean(latencies_fp32) / np.mean(latencies_fp16)
    }


# ============================================================================
# ONNX Runtime Benchmarking
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    input_size: Tuple[int, ...],
    onnx_path: str,
    opset_version: int = 13
) -> str:
    """Export PyTorch model to ONNX format."""
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(1, *input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    return onnx_path


def benchmark_onnx_runtime(
    onnx_path: str,
    input_size: Tuple[int, ...],
    num_warmup: int = 50,
    num_iterations: int = 200,
    batch_sizes: List[int] = [1, 8, 16, 32],
    providers: List[str] = None
) -> Dict:
    """
    Benchmark ONNX Runtime inference.
    
    Args:
        onnx_path: Path to ONNX model
        input_size: Input tensor size (C, H, W)
        num_warmup: Warmup iterations
        num_iterations: Measurement iterations
        batch_sizes: Batch sizes to test
        providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    Returns:
        Benchmark results dictionary
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {'error': 'onnxruntime not installed. Run: pip install onnxruntime-gpu'}
    
    # Determine available providers
    if providers is None:
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    
    results = {'providers': providers}
    
    # Create session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, session_options, providers=providers)
    input_name = session.get_inputs()[0].name
    
    for batch_size in batch_sizes:
        full_input_size = (batch_size,) + input_size
        test_input = np.random.randn(*full_input_size).astype(np.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = session.run(None, {input_name: test_input})
        
        # Measure
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: test_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        results[f'batch_{batch_size}'] = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'throughput_samples_per_sec': batch_size / (np.mean(latencies) / 1000)
        }
    
    return results


def benchmark_onnx_quantized(
    onnx_path: str,
    input_size: Tuple[int, ...],
    calibration_data: np.ndarray = None,
    num_warmup: int = 50,
    num_iterations: int = 200
) -> Dict:
    """
    Benchmark ONNX Runtime with quantization.
    """
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        return {'error': 'onnxruntime quantization not available'}
    
    # Quantize model
    quantized_path = onnx_path.replace('.onnx', '_quant.onnx')
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
    
    # Benchmark original
    results_fp32 = benchmark_onnx_runtime(
        onnx_path, input_size, num_warmup, num_iterations, [1]
    )
    
    # Benchmark quantized
    results_int8 = benchmark_onnx_runtime(
        quantized_path, input_size, num_warmup, num_iterations, [1]
    )
    
    return {
        'fp32': results_fp32.get('batch_1', {}),
        'int8': results_int8.get('batch_1', {}),
        'speedup': (results_fp32.get('batch_1', {}).get('mean_ms', 1) / 
                   max(results_int8.get('batch_1', {}).get('mean_ms', 1), 1e-6)),
        'model_sizes': {
            'fp32_mb': os.path.getsize(onnx_path) / (1024 * 1024),
            'int8_mb': os.path.getsize(quantized_path) / (1024 * 1024)
        }
    }


# ============================================================================
# TensorRT Benchmarking (NVIDIA GPUs)
# ============================================================================

def benchmark_tensorrt(
    onnx_path: str,
    input_size: Tuple[int, ...],
    num_warmup: int = 50,
    num_iterations: int = 200,
    batch_size: int = 1,
    fp16: bool = True
) -> Dict:
    """
    Benchmark TensorRT inference (NVIDIA GPUs only).
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        return {'error': 'TensorRT not available. Install tensorrt and pycuda.'}
    
    # Build TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            return {'error': 'Failed to parse ONNX model'}
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        return {'error': 'Failed to build TensorRT engine'}
    
    # Create execution context
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine)
    context = engine.create_execution_context()
    
    # Allocate buffers
    full_input_size = (batch_size,) + input_size
    input_mem = cuda.mem_alloc(np.prod(full_input_size) * 4)  # float32
    output_size = (batch_size, engine.get_binding_shape(1)[1])
    output_mem = cuda.mem_alloc(np.prod(output_size) * 4)
    
    stream = cuda.Stream()
    
    # Prepare input
    input_data = np.random.randn(*full_input_size).astype(np.float32)
    output_data = np.empty(output_size, dtype=np.float32)
    
    # Warmup
    for _ in range(num_warmup):
        cuda.memcpy_htod_async(input_mem, input_data, stream)
        context.execute_async_v2([int(input_mem), int(output_mem)], stream.handle)
        cuda.memcpy_dtoh_async(output_data, output_mem, stream)
        stream.synchronize()
    
    # Measure
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        cuda.memcpy_htod_async(input_mem, input_data, stream)
        context.execute_async_v2([int(input_mem), int(output_mem)], stream.handle)
        cuda.memcpy_dtoh_async(output_data, output_mem, stream)
        stream.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return {
        'precision': 'FP16' if fp16 else 'FP32',
        'batch_size': batch_size,
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'throughput_samples_per_sec': batch_size / (np.mean(latencies) / 1000)
    }


# ============================================================================
# Memory Profiling
# ============================================================================

def profile_memory(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu',
    batch_sizes: List[int] = [1, 8, 16, 32]
) -> Dict:
    """
    Profile memory usage during inference.
    """
    import tracemalloc
    
    model.eval()
    model = model.to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        gc.collect()
        
        full_input_size = (batch_size,) + input_size
        input_tensor = torch.randn(*full_input_size, device=device)
        
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            torch.cuda.synchronize()
            
            results[f'batch_{batch_size}'] = {
                'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'cached_memory_mb': torch.cuda.memory_reserved() / (1024 * 1024)
            }
        else:
            tracemalloc.start()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[f'batch_{batch_size}'] = {
                'current_memory_mb': current / (1024 * 1024),
                'peak_memory_mb': peak / (1024 * 1024)
            }
    
    return results


# ============================================================================
# Comprehensive Benchmark
# ============================================================================

def run_comprehensive_benchmark(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, ...],
    save_dir: str,
    include_onnx: bool = True,
    include_tensorrt: bool = False,
    batch_sizes: List[int] = [1, 8, 16, 32]
) -> Dict:
    """
    Run comprehensive benchmark across all available runtimes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'model_name': model_name,
        'input_size': input_size,
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info()
    }
    
    # Model complexity
    complexity = get_model_complexity(model)
    results['model_complexity'] = complexity
    
    # PyTorch CPU benchmark
    print("Benchmarking PyTorch CPU...")
    results['pytorch_cpu'] = benchmark_pytorch(
        model, input_size[1:], 'cpu', batch_sizes=batch_sizes
    )
    
    # PyTorch GPU benchmark
    if torch.cuda.is_available():
        print("Benchmarking PyTorch CUDA...")
        results['pytorch_cuda'] = benchmark_pytorch(
            model, input_size[1:], 'cuda', batch_sizes=batch_sizes
        )
        
        print("Benchmarking PyTorch AMP...")
        results['pytorch_amp'] = benchmark_pytorch_amp(
            model, input_size[1:]
        )
        
        # Memory profiling
        print("Profiling GPU memory...")
        results['gpu_memory'] = profile_memory(
            model, input_size[1:], 'cuda', batch_sizes
        )
    
    # ONNX Runtime benchmark
    if include_onnx:
        print("Exporting to ONNX...")
        onnx_path = os.path.join(save_dir, f'{model_name}.onnx')
        export_to_onnx(model, input_size[1:], onnx_path)
        
        print("Benchmarking ONNX Runtime...")
        results['onnx_runtime'] = benchmark_onnx_runtime(
            onnx_path, input_size[1:], batch_sizes=batch_sizes
        )
        
        print("Benchmarking ONNX Quantized...")
        results['onnx_quantized'] = benchmark_onnx_quantized(
            onnx_path, input_size[1:]
        )
    
    # TensorRT benchmark
    if include_tensorrt and torch.cuda.is_available():
        print("Benchmarking TensorRT...")
        onnx_path = os.path.join(save_dir, f'{model_name}.onnx')
        if not os.path.exists(onnx_path):
            export_to_onnx(model, input_size[1:], onnx_path)
        results['tensorrt'] = benchmark_tensorrt(onnx_path, input_size[1:])
    
    # CPU memory profiling
    print("Profiling CPU memory...")
    results['cpu_memory'] = profile_memory(model, input_size[1:], 'cpu', batch_sizes)
    
    # Save results
    results_path = os.path.join(save_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_benchmark_summary(results)
    
    return results


def get_system_info() -> Dict:
    """Get system information."""
    import platform
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def print_benchmark_summary(results: Dict):
    """Print benchmark summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\nModel: {results['model_name']}")
    print(f"Input Size: {results['input_size']}")
    
    complexity = results.get('model_complexity', {})
    print(f"Parameters: {complexity.get('params_m', 'N/A'):.2f}M")
    print(f"FLOPs: {complexity.get('flops_m', 'N/A'):.2f}M")
    
    print("\n" + "-"*80)
    print(f"{'Runtime':<25} {'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-"*80)
    
    # PyTorch CPU
    if 'pytorch_cpu' in results:
        for batch_key, metrics in results['pytorch_cpu'].items():
            if batch_key.startswith('batch_'):
                batch_size = batch_key.split('_')[1]
                print(f"{'PyTorch CPU':<25} {batch_size:<8} "
                      f"{metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f}    "
                      f"{metrics['throughput_samples_per_sec']:.1f} samples/s")
    
    # PyTorch CUDA
    if 'pytorch_cuda' in results:
        for batch_key, metrics in results['pytorch_cuda'].items():
            if batch_key.startswith('batch_'):
                batch_size = batch_key.split('_')[1]
                print(f"{'PyTorch CUDA':<25} {batch_size:<8} "
                      f"{metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f}    "
                      f"{metrics['throughput_samples_per_sec']:.1f} samples/s")
    
    # ONNX Runtime
    if 'onnx_runtime' in results and 'error' not in results['onnx_runtime']:
        for batch_key, metrics in results['onnx_runtime'].items():
            if batch_key.startswith('batch_'):
                batch_size = batch_key.split('_')[1]
                print(f"{'ONNX Runtime':<25} {batch_size:<8} "
                      f"{metrics['mean_ms']:.2f} ± {metrics['std_ms']:.2f}    "
                      f"{metrics['throughput_samples_per_sec']:.1f} samples/s")
    
    # ONNX Quantized
    if 'onnx_quantized' in results and 'error' not in results['onnx_quantized']:
        int8 = results['onnx_quantized'].get('int8', {})
        if int8:
            print(f"{'ONNX INT8':<25} {'1':<8} "
                  f"{int8['mean_ms']:.2f} ± {int8['std_ms']:.2f}    "
                  f"{1000/int8['mean_ms']:.1f} samples/s")
    
    print("-"*80)


# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, ...],
    save_dir: str,
    batch_size: int = 1
) -> Dict:
    """
    Compare multiple models on the same benchmark.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    comparison = {}
    
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}...")
        
        complexity = get_model_complexity(model)
        
        # CPU benchmark
        cpu_results = benchmark_pytorch(
            model, input_size[1:], 'cpu', batch_sizes=[batch_size]
        )
        
        # GPU benchmark
        gpu_results = {}
        if torch.cuda.is_available():
            gpu_results = benchmark_pytorch(
                model, input_size[1:], 'cuda', batch_sizes=[batch_size]
            )
        
        comparison[model_name] = {
            'params_m': complexity['params_m'],
            'flops_m': complexity['flops_m'],
            'cpu_latency_ms': cpu_results.get(f'batch_{batch_size}', {}).get('mean_ms', 0),
            'cpu_throughput': cpu_results.get(f'batch_{batch_size}', {}).get('throughput_samples_per_sec', 0),
            'gpu_latency_ms': gpu_results.get(f'batch_{batch_size}', {}).get('mean_ms', 0),
            'gpu_throughput': gpu_results.get(f'batch_{batch_size}', {}).get('throughput_samples_per_sec', 0)
        }
    
    # Save comparison
    comparison_path = os.path.join(save_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<25} {'Params':<10} {'FLOPs':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'GPU Thrpt':<15}")
    print("-"*100)
    
    for model_name, metrics in comparison.items():
        print(f"{model_name:<25} {metrics['params_m']:<10.2f} {metrics['flops_m']:<10.2f} "
              f"{metrics['cpu_latency_ms']:<12.2f} {metrics['gpu_latency_ms']:<12.2f} "
              f"{metrics['gpu_throughput']:<15.1f}")
    
    return comparison


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Laptop/Desktop Benchmark')
    
    parser.add_argument('--model', type=str, default='ghost_resnet18',
                       help='Model to benchmark')
    parser.add_argument('--attention', type=str, default='coordinate',
                       choices=['none', 'se', 'cbam', 'eca', 'coord', 'coordinate'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--weights', type=str, default='weights/baseline.pth',
                       help='Path to model weights')
    parser.add_argument('--input_size', type=str, default='1,3,32,32',
                       help='Input size')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 16, 32],
                       help='Batch sizes to test')
    parser.add_argument('--save_dir', type=str, default='weights/benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--include_onnx', action='store_true', default=False,
                       help='Include ONNX Runtime benchmark (disabled by default due to CoordAtt compatibility)')
    parser.add_argument('--include_tensorrt', action='store_true', default=False,
                       help='Include TensorRT benchmark')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='Models to compare')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*60)
    print("Laptop/Desktop Benchmark Suite")
    print("="*60)
    
    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))
    
    if args.compare:
        # Compare multiple models
        models = {}
        for model_name in args.compare:
            models[model_name] = create_model(
                model_name, 
                num_classes=args.num_classes,
                attention_type=args.attention
            )
        
        compare_models(models, input_size, args.save_dir)
    else:
        # Single model benchmark
        model = create_model(
            args.model,
            num_classes=args.num_classes,
            attention_type=args.attention
        )
        
        if args.weights and os.path.exists(args.weights):
            checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # Filter out profiling keys
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.endswith(('total_ops', 'total_params'))}
            model.load_state_dict(state_dict, strict=False)
        
        run_comprehensive_benchmark(
            model=model,
            model_name=args.model,
            input_size=input_size,
            save_dir=args.save_dir,
            include_onnx=args.include_onnx,
            include_tensorrt=args.include_tensorrt,
            batch_sizes=args.batch_sizes
        )