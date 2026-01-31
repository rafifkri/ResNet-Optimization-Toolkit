"""
===================================================================================
Model Factory - Unified Model Builder
===================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from models.resnet_base import (
    resnet18_base, resnet34_base, resnet50_base,
    resnet18_imagenet, resnet50_imagenet, resnet50_teacher
)
from models.resnet_ghost import (
    ghost_resnet18, ghost_resnet34, ghost_resnet50,
    ghost_resnet18_imagenet, ghost_resnet_small, ghost_resnet_tiny
)


# ===================================================================================
# MODEL REGISTRY
# ===================================================================================

MODEL_REGISTRY = {
    # Baseline ResNet (CIFAR)
    "resnet18": resnet18_base,
    "resnet18_base": resnet18_base,
    "resnet34": resnet34_base,
    "resnet34_base": resnet34_base,
    "resnet50": resnet50_base,
    "resnet50_base": resnet50_base,
    
    # Baseline ResNet (ImageNet)
    "resnet18_imagenet": resnet18_imagenet,
    "resnet50_imagenet": resnet50_imagenet,
    
    # Teacher model
    "resnet50_teacher": resnet50_teacher,
    
    # Ghost-ResNet (CIFAR)
    "ghost_resnet18": ghost_resnet18,
    "ghost_resnet34": ghost_resnet34,
    "ghost_resnet50": ghost_resnet50,
    
    # Ghost-ResNet (ImageNet)
    "ghost_resnet18_imagenet": ghost_resnet18_imagenet,
    
    # Compact variants
    "ghost_resnet_small": ghost_resnet_small,
    "ghost_resnet_tiny": ghost_resnet_tiny,
}


# ===================================================================================
# MODEL BUILDER
# ===================================================================================

def create_model(model_name: str, 
                 num_classes: int = 10,
                 pretrained: bool = False,
                 checkpoint_path: Optional[str] = None,
                 **kwargs) -> nn.Module:
    """
    Create a model by name.
    
    Args:
        model_name: Name of the model (see MODEL_REGISTRY)
        num_classes: Number of output classes
        pretrained: Load pretrained weights (for torchvision models)
        checkpoint_path: Path to custom checkpoint
        **kwargs: Additional arguments for model constructor
    
    Returns:
        nn.Module: The created model
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(num_classes=num_classes, **kwargs)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Handle DataParallel checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    
    return model


def get_model_info(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> Dict[str, Any]:
    """
    Get model information (parameters, FLOPs, etc.)
    
    Args:
        model: The model to analyze
        input_size: Input tensor size
    
    Returns:
        Dictionary with model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Try to compute FLOPs
    flops = None
    try:
        from thop import profile
        # Get device from model parameters
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except (ImportError, StopIteration):
        pass
    except Exception:
        pass  # Ignore profiling errors
    
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
    }
    
    if flops is not None:
        info["flops"] = flops
        info["flops_m"] = flops / 1e6
        info["flops_g"] = flops / 1e9
    
    return info


def list_models() -> list:
    """List all available models"""
    return list(MODEL_REGISTRY.keys())


# Alias for backward compatibility
get_available_models = list_models


# ===================================================================================
# TESTING
# ===================================================================================

if __name__ == "__main__":
    print("Available models:")
    for name in list_models():
        print(f"  - {name}")
    
    print("\nTesting model creation...")
    for name in ["resnet18", "ghost_resnet18", "ghost_resnet_tiny"]:
        model = create_model(name, num_classes=10)
        info = get_model_info(model)
        print(f"{name}: {info['total_params_m']:.2f}M params", end="")
        if 'flops_m' in info:
            print(f", {info['flops_m']:.2f}M FLOPs")
        else:
            print()
