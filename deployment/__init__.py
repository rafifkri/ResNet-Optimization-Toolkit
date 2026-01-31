"""
deployment/__init__.py

Deployment utilities for ResNet optimization research.
"""

from deployment.android_export import (
    export_torchscript,
    export_mobile_optimized,
    export_onnx,
    export_for_android
)

from deployment.laptop_bench import (
    benchmark_pytorch,
    benchmark_onnx_runtime,
    run_comprehensive_benchmark,
    compare_models
)

__all__ = [
    # Android/Mobile export
    'export_torchscript',
    'export_mobile_optimized',
    'export_onnx',
    'export_for_android',
    
    # Benchmarking
    'benchmark_pytorch',
    'benchmark_onnx_runtime',
    'run_comprehensive_benchmark',
    'compare_models'
]
