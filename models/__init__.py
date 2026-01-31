"""
===================================================================================
Models Package
===================================================================================
"""

from .resnet_base import (
    ResNet, BasicBlock, Bottleneck,
    resnet18_base, resnet34_base, resnet50_base,
    resnet18_imagenet, resnet50_imagenet, resnet50_teacher
)

from .resnet_ghost import (
    GhostResNet, GhostBasicBlock, GhostBottleneckBlock,
    ghost_resnet18, ghost_resnet34, ghost_resnet50,
    ghost_resnet18_imagenet, ghost_resnet_small, ghost_resnet_tiny
)

from .model_factory import (
    create_model, get_model_info, list_models, get_available_models, MODEL_REGISTRY
)

__all__ = [
    # Base ResNet
    'ResNet', 'BasicBlock', 'Bottleneck',
    'resnet18_base', 'resnet34_base', 'resnet50_base',
    'resnet18_imagenet', 'resnet50_imagenet', 'resnet50_teacher',
    # Ghost ResNet
    'GhostResNet', 'GhostBasicBlock', 'GhostBottleneckBlock',
    'ghost_resnet18', 'ghost_resnet34', 'ghost_resnet50',
    'ghost_resnet18_imagenet', 'ghost_resnet_small', 'ghost_resnet_tiny',
    # Factory
    'create_model', 'get_model_info', 'list_models', 'get_available_models', 'MODEL_REGISTRY',
]
