"""
===================================================================================
Ghost-ResNet Implementation
===================================================================================
Combines ResNet architecture with Ghost Modules and Attention mechanisms
for efficient inference with minimal accuracy loss.
===================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional, Union
import math

from models.layers.ghost_module import GhostModule, GhostBottleneck
from models.layers.attention import get_attention, CoordinateAttention
from models.layers.drop_path import DropPath


# ===================================================================================
# GHOST BASIC BLOCK
# ===================================================================================

class GhostBasicBlock(nn.Module):
    """
    Ghost Basic Block for ResNet-18/34 style networks.
    
    Replaces standard convolutions with Ghost Modules for efficiency.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attention_type: str = "coordinate", attention_reduction: int = 32,
                 ghost_ratio: int = 2, drop_path_rate: float = 0.0):
        super(GhostBasicBlock, self).__init__()
        
        # Replace Conv with Ghost Module
        self.ghost1 = GhostModule(in_planes, planes, kernel_size=3, stride=stride, ratio=ghost_ratio)
        self.ghost2 = GhostModule(planes, planes, kernel_size=3, ratio=ghost_ratio, relu=False)
        
        # Attention module
        self.attention = get_attention(attention_type, planes, attention_reduction)
        
        # Shortcut connection
        self.downsample = downsample
        if downsample is None and (stride != 1 or in_planes != planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        # Drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.ghost1(x)
        out = self.ghost2(out)
        out = self.attention(out)
        
        out = self.drop_path(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)


class GhostBottleneckBlock(nn.Module):
    """
    Ghost Bottleneck Block for ResNet-50/101 style networks.
    
    Uses Ghost Modules in the expansion/projection stages.
    """
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attention_type: str = "coordinate", attention_reduction: int = 32,
                 ghost_ratio: int = 2, drop_path_rate: float = 0.0):
        super(GhostBottleneckBlock, self).__init__()
        
        # 1x1 reduce
        self.ghost1 = GhostModule(in_planes, planes, kernel_size=1, ratio=ghost_ratio)
        
        # 3x3 depthwise
        self.conv_dw = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                  padding=1, groups=planes, bias=False)
        self.bn_dw = nn.BatchNorm2d(planes)
        
        # 1x1 expand
        self.ghost2 = GhostModule(planes, planes * self.expansion, kernel_size=1, 
                                   ratio=ghost_ratio, relu=False)
        
        # Attention
        self.attention = get_attention(attention_type, planes * self.expansion, attention_reduction)
        
        # Shortcut
        self.downsample = downsample
        if downsample is None and (stride != 1 or in_planes != planes * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        
        # Drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.ghost1(x)
        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = F.relu(out)
        out = self.ghost2(out)
        out = self.attention(out)
        
        out = self.drop_path(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)


# ===================================================================================
# GHOST RESNET
# ===================================================================================

class GhostResNet(nn.Module):
    """
    Ghost-ResNet: Efficient ResNet with Ghost Modules and Attention.
    
    Args:
        block: Block type (GhostBasicBlock or GhostBottleneckBlock)
        layers: Number of blocks per layer
        num_classes: Number of output classes
        width_multiplier: Channel width multiplier
        attention_type: Attention type ('coordinate', 'se', 'cbam', 'eca', 'none')
        attention_reduction: Attention reduction ratio
        ghost_ratio: Ghost module ratio (higher = more efficient)
        drop_path_rate: Stochastic depth rate
        dropout_rate: Classifier dropout rate
        cifar: Use CIFAR-style stem
    """
    def __init__(self,
                 block: Type[Union[GhostBasicBlock, GhostBottleneckBlock]],
                 layers: List[int],
                 num_classes: int = 10,
                 width_multiplier: float = 1.0,
                 attention_type: str = "coordinate",
                 attention_reduction: int = 32,
                 ghost_ratio: int = 2,
                 drop_path_rate: float = 0.0,
                 dropout_rate: float = 0.0,
                 cifar: bool = True):
        super(GhostResNet, self).__init__()
        
        self.in_planes = int(64 * width_multiplier)
        self.attention_type = attention_type
        self.attention_reduction = attention_reduction
        self.ghost_ratio = ghost_ratio
        
        # Calculate drop path rates
        total_blocks = sum(layers)
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.block_idx = 0
        
        # Stem
        if cifar:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, 
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.maxpool = None
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), layers[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(int(512 * width_multiplier) * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            self.attention_type, self.attention_reduction,
            self.ghost_ratio, self.drop_path_rates[self.block_idx]
        ))
        self.block_idx += 1
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.in_planes, planes,
                attention_type=self.attention_type,
                attention_reduction=self.attention_reduction,
                ghost_ratio=self.ghost_ratio,
                drop_path_rate=self.drop_path_rates[self.block_idx]
            ))
            self.block_idx += 1

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features for distillation"""
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features


# ===================================================================================
# MODEL CONSTRUCTORS
# ===================================================================================

def ghost_resnet18(num_classes: int = 10, **kwargs) -> GhostResNet:
    """Ghost-ResNet-18 for CIFAR"""
    return GhostResNet(GhostBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def ghost_resnet34(num_classes: int = 10, **kwargs) -> GhostResNet:
    """Ghost-ResNet-34 for CIFAR"""
    return GhostResNet(GhostBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ghost_resnet50(num_classes: int = 10, **kwargs) -> GhostResNet:
    """Ghost-ResNet-50 for CIFAR"""
    return GhostResNet(GhostBottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ghost_resnet18_imagenet(num_classes: int = 1000, **kwargs) -> GhostResNet:
    """Ghost-ResNet-18 for ImageNet"""
    return GhostResNet(GhostBasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=False, **kwargs)

# Convenience aliases
def ghost_resnet_small(num_classes: int = 10, **kwargs) -> GhostResNet:
    """Smaller Ghost-ResNet with width multiplier 0.5"""
    return ghost_resnet18(num_classes=num_classes, width_multiplier=0.5, **kwargs)

def ghost_resnet_tiny(num_classes: int = 10, **kwargs) -> GhostResNet:
    """Tiny Ghost-ResNet with width multiplier 0.25"""
    return ghost_resnet18(num_classes=num_classes, width_multiplier=0.25, **kwargs)


# ===================================================================================
# TESTING
# ===================================================================================

if __name__ == "__main__":
    # Test models
    x = torch.randn(2, 3, 32, 32)
    
    print("Testing Ghost-ResNet variants...")
    
    for name, model_fn in [
        ("Ghost-ResNet-18", ghost_resnet18),
        ("Ghost-ResNet-34", ghost_resnet34),
        ("Ghost-ResNet-50", ghost_resnet50),
        ("Ghost-ResNet-Small", ghost_resnet_small),
        ("Ghost-ResNet-Tiny", ghost_resnet_tiny),
    ]:
        model = model_fn(num_classes=10)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name}: Output {out.shape}, Params: {params:.2f}M")
    
    # Test different attention types
    print("\nTesting different attention types...")
    for att in ["none", "se", "cbam", "eca", "coordinate"]:
        model = ghost_resnet18(num_classes=10, attention_type=att)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Attention={att}: Params: {params:.2f}M")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    model = ghost_resnet18(num_classes=10)
    features = model.get_features(x)
    for i, f in enumerate(features):
        print(f"Layer {i+1}: {f.shape}")