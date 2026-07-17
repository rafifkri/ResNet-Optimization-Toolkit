import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional, Union
from models.layers.attention import get_attention
from models.layers.drop_path import DropPath

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attention_type: str = "none", attention_reduction: int = 16,
                 drop_path_rate: float = 0.0):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
        # Attention module
        self.attention = get_attention(attention_type, planes, attention_reduction)
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Apply drop path
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attention_type: str = "none", attention_reduction: int = 16,
                 drop_path_rate: float = 0.0):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Attention module
        self.attention = get_attention(attention_type, planes * self.expansion, attention_reduction)
        
        # Drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Apply drop path
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]], 
                 layers: List[int],
                 num_classes: int = 10,
                 width_multiplier: float = 1.0,
                 attention_type: str = "none",
                 attention_reduction: int = 16,
                 drop_path_rate: float = 0.0,
                 dropout_rate: float = 0.0,
                 zero_init_residual: bool = True,
                 cifar: bool = True):
        super(ResNet, self).__init__()
        
        self.in_planes = int(64 * width_multiplier)
        self.attention_type = attention_type
        self.attention_reduction = attention_reduction
        self.drop_path_rate = drop_path_rate
        
        # Calculate drop path rates for each block
        total_blocks = sum(layers)
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.block_idx = 0
        
        # Stem
        if cifar:
            # CIFAR-style stem (32x32 input)
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, 
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.maxpool = None
        else:
            # ImageNet-style stem (224x224 input)
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), layers[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(int(512 * width_multiplier) * block.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], 
                    planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            self.attention_type, self.attention_reduction,
            self.drop_path_rates[self.block_idx]
        ))
        self.block_idx += 1
        self.in_planes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(
                self.in_planes, planes,
                attention_type=self.attention_type,
                attention_reduction=self.attention_reduction,
                drop_path_rate=self.drop_path_rates[self.block_idx]
            ))
            self.block_idx += 1

        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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
        """Extract intermediate features (useful for distillation)"""
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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

def resnet18_base(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-18 for CIFAR"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=True, **kwargs)

def resnet34_base(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-34 for CIFAR"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar=True, **kwargs)

def resnet50_base(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-50 for CIFAR"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cifar=True, **kwargs)

def resnet18_imagenet(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-18 for ImageNet"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar=False, **kwargs)

def resnet50_imagenet(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-50 for ImageNet"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cifar=False, **kwargs)

def resnet50_teacher(num_classes: int = 10):
    """ResNet-50 as teacher model (using torchvision pretrained)"""
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    except:
        return resnet50_base(num_classes=num_classes)

if __name__ == "__main__":
    # Test models
    x = torch.randn(2, 3, 32, 32)  # CIFAR input
    
    print("Testing ResNet variants...")
    
    for name, model_fn in [
        ("ResNet-18", resnet18_base),
        ("ResNet-34", resnet34_base),
        ("ResNet-50", resnet50_base),
    ]:
        model = model_fn(num_classes=10)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name}: Output {out.shape}, Params: {params:.2f}M")
    
    # Test with attention
    print("\nTesting with Coordinate Attention...")
    model = resnet18_base(num_classes=10, attention_type="coordinate")
    out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ResNet-18 + CA: Output {out.shape}, Params: {params:.2f}M")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = model.get_features(x)
    for i, f in enumerate(features):
        print(f"Layer {i+1}: {f.shape}")