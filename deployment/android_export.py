"""
android_export.py - Export Models for Android/Mobile Deployment

This script provides comprehensive model export functionality for mobile devices:
1. TorchScript (JIT) export
2. PyTorch Mobile optimization
3. TensorFlow Lite conversion (via ONNX)
4. Model size optimization
5. Quantized mobile export

Reference:
- PyTorch Mobile: https://pytorch.org/mobile
- TensorFlow Lite: https://www.tensorflow.org/lite
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import (
    optimize_for_mobile,
    MobileOptimizerType
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model
from utils.metrics import get_model_complexity, measure_latency


# ============================================================================
# TorchScript Export
# ============================================================================

def export_torchscript(
    model: nn.Module,
    example_input: torch.Tensor,
    save_path: str,
    method: str = 'trace'
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor for tracing
        save_path: Path to save the exported model
        method: 'trace' or 'script'
    
    Returns:
        Path to saved model
    """
    model.eval()
    model.cpu()
    example_input = example_input.cpu()
    
    if method == 'trace':
        scripted_model = torch.jit.trace(model, example_input)
    elif method == 'script':
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scripted_model.save(save_path)
    
    return save_path


def export_mobile_optimized(
    model: nn.Module,
    example_input: torch.Tensor,
    save_path: str,
    optimization_blocklist: List[MobileOptimizerType] = None
) -> str:
    """
    Export model optimized for mobile inference.
    
    Applies optimizations:
    - Fuses Conv-BN-ReLU
    - Removes dropout
    - Applies memory optimization
    
    Args:
        model: PyTorch model
        example_input: Example input for tracing
        save_path: Path to save optimized model
        optimization_blocklist: List of optimizations to skip
    
    Returns:
        Path to saved model
    """
    model.eval()
    model.cpu()
    example_input = example_input.cpu()
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    if optimization_blocklist:
        optimized_model = optimize_for_mobile(
            traced_model, 
            optimization_blocklist=set(optimization_blocklist)
        )
    else:
        optimized_model = optimize_for_mobile(traced_model)
    
    # Save for lite interpreter (Android/iOS)
    optimized_model._save_for_lite_interpreter(save_path)
    
    return save_path


def export_quantized_mobile(
    model: nn.Module,
    calibration_data: torch.Tensor,
    save_path: str,
    backend: str = 'qnnpack'
) -> str:
    """
    Export quantized model for mobile.
    
    Uses dynamic quantization optimized for mobile inference.
    
    Args:
        model: PyTorch model
        calibration_data: Data for calibration
        save_path: Path to save model
        backend: Quantization backend ('qnnpack' for mobile)
    
    Returns:
        Path to saved model
    """
    model.eval()
    model.cpu()
    
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Dynamic quantization (simpler, works better on mobile)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    # Trace and optimize
    example_input = calibration_data[0:1].cpu()
    traced_model = torch.jit.trace(quantized_model, example_input)
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save for lite interpreter
    optimized_model._save_for_lite_interpreter(save_path)
    
    return save_path


# ============================================================================
# ONNX Export
# ============================================================================

def export_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    save_path: str,
    opset_version: int = 13,
    dynamic_axes: bool = True
) -> str:
    """
    Export model to ONNX format.
    
    ONNX can be converted to TensorFlow Lite or other formats.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor
        save_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
    
    Returns:
        Path to saved model
    """
    model.eval()
    model.cpu()
    example_input = example_input.cpu()
    
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        example_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config
    )
    
    return save_path


def verify_onnx(onnx_path: str) -> bool:
    """Verify ONNX model is valid."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def simplify_onnx(onnx_path: str, output_path: str = None) -> str:
    """Simplify ONNX model using onnx-simplifier."""
    try:
        import onnx
        from onnxsim import simplify
        
        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)
        
        if output_path is None:
            output_path = onnx_path.replace('.onnx', '_simplified.onnx')
        
        onnx.save(model_simplified, output_path)
        return output_path
    except ImportError:
        print("onnx-simplifier not installed. Run: pip install onnx-simplifier")
        return onnx_path


# ============================================================================
# TensorFlow Lite Conversion
# ============================================================================

def convert_onnx_to_tflite(
    onnx_path: str,
    save_path: str,
    quantize: bool = False,
    representative_dataset = None
) -> str:
    """
    Convert ONNX model to TensorFlow Lite.
    
    Args:
        onnx_path: Path to ONNX model
        save_path: Path to save TFLite model
        quantize: Apply INT8 quantization
        representative_dataset: Function returning calibration data
    
    Returns:
        Path to saved TFLite model
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel
        saved_model_path = save_path.replace('.tflite', '_savedmodel')
        tf_rep.export_graph(saved_model_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        return save_path
        
    except ImportError as e:
        print(f"TFLite conversion requires: pip install onnx-tf tensorflow")
        print(f"Error: {e}")
        return None


# ============================================================================
# Model Analysis
# ============================================================================

def analyze_mobile_model(model_path: str, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> Dict:
    """
    Analyze exported mobile model.
    
    Args:
        model_path: Path to exported model
        input_size: Input tensor size
    
    Returns:
        Dictionary with model metrics
    """
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    analysis = {
        'path': model_path,
        'file_size_mb': file_size_mb,
        'format': Path(model_path).suffix
    }
    
    # Load and measure latency for TorchScript models
    if model_path.endswith('.pt') or model_path.endswith('.ptl'):
        try:
            if model_path.endswith('.ptl'):
                model = torch.jit.load(model_path)
            else:
                model = torch.jit.load(model_path)
            
            model.eval()
            model.cpu()
            
            # Measure latency
            import time
            input_tensor = torch.randn(*input_size)
            
            # Warmup
            for _ in range(10):
                _ = model(input_tensor)
            
            # Measure
            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            analysis['latency_ms'] = {
                'mean': sum(times) / len(times),
                'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
                'min': min(times),
                'max': max(times)
            }
        except Exception as e:
            analysis['error'] = str(e)
    
    # Analyze ONNX models
    elif model_path.endswith('.onnx'):
        try:
            import onnx
            model = onnx.load(model_path)
            
            # Get input/output info
            analysis['inputs'] = [
                {'name': inp.name, 'shape': [d.dim_value for d in inp.type.tensor_type.shape.dim]}
                for inp in model.graph.input
            ]
            analysis['outputs'] = [
                {'name': out.name, 'shape': [d.dim_value for d in out.type.tensor_type.shape.dim]}
                for out in model.graph.output
            ]
            
            # Count operations
            analysis['num_nodes'] = len(model.graph.node)
            
        except Exception as e:
            analysis['error'] = str(e)
    
    return analysis


def generate_android_code(
    model_path: str,
    class_names: List[str],
    output_dir: str
) -> str:
    """
    Generate sample Android/Kotlin code for model inference.
    
    Args:
        model_path: Path to the mobile model
        class_names: List of class names
        output_dir: Directory to save generated code
    
    Returns:
        Path to generated code file
    """
    model_name = Path(model_path).stem
    
    kotlin_code = f'''
package com.example.imagerecognition

import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File

/**
 * Image Classification using {model_name}
 * 
 * Generated by ResNet Optimization Research Framework
 */
class ImageClassifier(private val modelPath: String) {{
    
    private var module: Module? = null
    
    private val classNames = listOf(
        {', '.join(f'"{name}"' for name in class_names)}
    )
    
    init {{
        module = Module.load(modelPath)
    }}
    
    /**
     * Classify an image
     * @param bitmap Input image as Bitmap
     * @return Pair of (class name, confidence score)
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {{
        // Preprocess image
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            floatArrayOf(0.485f, 0.456f, 0.406f),  // mean
            floatArrayOf(0.229f, 0.224f, 0.225f)   // std
        )
        
        // Run inference
        val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
        
        // Get scores
        val scores = outputTensor?.dataAsFloatArray ?: return Pair("Unknown", 0f)
        
        // Softmax
        val expScores = scores.map {{ Math.exp(it.toDouble()).toFloat() }}
        val sumExp = expScores.sum()
        val probabilities = expScores.map {{ it / sumExp }}
        
        // Find max
        val maxIdx = probabilities.indices.maxByOrNull {{ probabilities[it] }} ?: 0
        
        return Pair(classNames[maxIdx], probabilities[maxIdx])
    }}
    
    /**
     * Get top-k predictions
     */
    fun classifyTopK(bitmap: Bitmap, k: Int = 5): List<Pair<String, Float>> {{
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            floatArrayOf(0.485f, 0.456f, 0.406f),
            floatArrayOf(0.229f, 0.224f, 0.225f)
        )
        
        val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
        val scores = outputTensor?.dataAsFloatArray ?: return emptyList()
        
        // Softmax
        val expScores = scores.map {{ Math.exp(it.toDouble()).toFloat() }}
        val sumExp = expScores.sum()
        val probabilities = expScores.map {{ it / sumExp }}
        
        // Top-k
        return probabilities.indices
            .sortedByDescending {{ probabilities[it] }}
            .take(k)
            .map {{ Pair(classNames[it], probabilities[it]) }}
    }}
    
    fun release() {{
        module?.destroy()
        module = null
    }}
}}

// Usage example in Activity:
// val classifier = ImageClassifier(assetFilePath(this, "{Path(model_path).name}"))
// val (className, confidence) = classifier.classify(bitmap)
'''
    
    os.makedirs(output_dir, exist_ok=True)
    code_path = os.path.join(output_dir, 'ImageClassifier.kt')
    
    with open(code_path, 'w') as f:
        f.write(kotlin_code)
    
    return code_path


# ============================================================================
# Main Export Function
# ============================================================================

def export_for_android(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, ...],
    save_dir: str,
    class_names: List[str] = None,
    quantize: bool = True,
    generate_code: bool = True
) -> Dict:
    """
    Complete export pipeline for Android deployment.
    
    Args:
        model: PyTorch model
        model_name: Name for exported files
        input_size: Input tensor size
        save_dir: Directory to save exports
        class_names: List of class names
        quantize: Export quantized version
        generate_code: Generate sample Android code
    
    Returns:
        Dictionary with paths to exported files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    example_input = torch.randn(*input_size)
    
    exports = {}
    
    # 1. TorchScript (standard)
    print("Exporting TorchScript...")
    ts_path = os.path.join(save_dir, f'{model_name}.pt')
    export_torchscript(model, example_input, ts_path)
    exports['torchscript'] = ts_path
    
    # 2. Mobile optimized
    print("Exporting Mobile-optimized model...")
    mobile_path = os.path.join(save_dir, f'{model_name}_mobile.ptl')
    export_mobile_optimized(model, example_input, mobile_path)
    exports['mobile'] = mobile_path
    
    # 3. Quantized mobile (optional)
    if quantize:
        print("Exporting Quantized mobile model...")
        quant_path = os.path.join(save_dir, f'{model_name}_mobile_quant.ptl')
        try:
            export_quantized_mobile(model, example_input.expand(32, -1, -1, -1), quant_path)
            exports['quantized_mobile'] = quant_path
        except Exception as e:
            print(f"Quantized export failed: {e}")
    
    # 4. ONNX (for cross-platform) - Skip for CoordinateAttention models
    print("Skipping ONNX export (CoordinateAttention not compatible with ONNX)...")
    # onnx_path = os.path.join(save_dir, f'{model_name}.onnx')
    # export_onnx(model, example_input, onnx_path)
    # exports['onnx'] = onnx_path
    
    # # Verify ONNX
    # if verify_onnx(onnx_path):
    #     print("ONNX model verified successfully")
    
    # 5. Generate sample code
    if generate_code and class_names:
        print("Generating sample Android code...")
        code_path = generate_android_code(mobile_path, class_names, 
                                         os.path.join(save_dir, 'android_code'))
        exports['android_code'] = code_path
    
    # 6. Analyze all exports
    print("\nAnalyzing exports...")
    analysis = {}
    for name, path in exports.items():
        if path and os.path.exists(path) and not path.endswith('.kt'):
            analysis[name] = analyze_mobile_model(path, input_size)
    
    # Save analysis
    analysis_path = os.path.join(save_dir, 'export_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    exports['analysis'] = analysis_path
    
    # Print summary
    print("\n" + "="*50)
    print("Export Summary")
    print("="*50)
    for name, info in analysis.items():
        print(f"{name}: {info['file_size_mb']:.2f} MB")
        if 'latency_ms' in info:
            print(f"  Latency: {info['latency_ms']['mean']:.2f} ms")
    
    return exports


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Export models for Android/Mobile')
    
    parser.add_argument('--model', type=str, default='ghost_resnet18',
                       help='Model architecture')
    parser.add_argument('--attention', type=str, default='coordinate',
                       choices=['none', 'se', 'cbam', 'eca', 'coord', 'coordinate'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--weights', type=str, default='weights/baseline.pth',
                       help='Path to model weights')
    parser.add_argument('--input_size', type=str, default='1,3,32,32',
                       help='Input size (batch,channels,height,width)')
    parser.add_argument('--save_dir', type=str, default='weights/mobile_export',
                       help='Directory to save exports')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Export quantized version')
    parser.add_argument('--generate_code', action='store_true', default=True,
                       help='Generate sample Android code')
    parser.add_argument('--class_names', type=str, nargs='+', 
                       default=['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck'],
                       help='Class names for code generation')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*50)
    print("Android/Mobile Model Export")
    print("="*50)
    
    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))
    
    # Create model
    model = create_model(args.model, num_classes=args.num_classes,
                        attention_type=args.attention)
    
    # Load weights
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Filter profiling keys
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.endswith(('total_ops', 'total_params'))}
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Weights not found at {args.weights}")
    
    # Export
    exports = export_for_android(
        model=model,
        model_name=args.model,
        input_size=input_size,
        save_dir=args.save_dir,
        class_names=args.class_names,
        quantize=args.quantize,
        generate_code=args.generate_code
    )
    
    print(f"\nExports saved to {args.save_dir}")
    print("Files:")
    for name, path in exports.items():
        if path:
            print(f"  - {name}: {path}")