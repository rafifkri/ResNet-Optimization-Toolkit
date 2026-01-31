"""
===================================================================================
Ghost Module Implementation
===================================================================================
Reference: GhostNet: More Features from Cheap Operations (CVPR 2020)
Paper: https://arxiv.org/abs/1911.11907
===================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GhostModule(nn.Module):
    """
    Ghost Module: Generate more features from cheap operations.
    
    Ghost modules split convolution into two parts:
    1. Primary convolution: Standard convolution to generate intrinsic features
    2. Cheap operation: Depthwise convolution to generate ghost features
    
    This reduces computation while maintaining representation capacity.
    
    Args:
        inp: Number of input channels
        oup: Number of output channels  
        kernel_size: Kernel size for primary convolution
        ratio: Ratio of ghost features to intrinsic features
        dw_size: Kernel size for depthwise (cheap) operation
        stride: Stride for primary convolution
        relu: Whether to apply ReLU activation
    """
    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2, 
                 dw_size: int = 3, stride: int = 1, relu: bool = True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # Cheap operation (ghost features via depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, 
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck with optional SE attention.
    
    Similar to MobileNetV3 bottleneck but uses Ghost modules.
    
    Args:
        in_chs: Input channels
        mid_chs: Middle (expansion) channels
        out_chs: Output channels
        dw_kernel_size: Depthwise conv kernel size
        stride: Stride for downsampling
        se_ratio: SE ratio (0 to disable)
    """
    def __init__(self, in_chs: int, mid_chs: int, out_chs: int, 
                 dw_kernel_size: int = 3, stride: int = 1, se_ratio: float = 0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size-1)//2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # Shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                         padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class SqueezeExcite(nn.Module):
    """SE module for GhostBottleneck"""
    def __init__(self, in_chs: int, se_ratio: float = 0.25, 
                 reduced_base_chs: Optional[int] = None):
        super(SqueezeExcite, self).__init__()
        reduced_chs = max(1, int(in_chs * se_ratio))
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * x_se.sigmoid()


class GhostModuleV2(nn.Module):
    """
    GhostNetV2 Module with DFC (Decoupled Fully Connected) attention.
    Reference: GhostNetV2: Enhance Cheap Operation with Long-Range Attention
    
    Args:
        inp: Input channels
        oup: Output channels
        kernel_size: Kernel size
        ratio: Ghost ratio
        dw_size: Depthwise kernel size
        stride: Stride
        relu: Whether to use ReLU
        mode: 'original' for GhostV1, 'attn' for GhostV2 with DFC attention
    """
    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2,
                 dw_size: int = 3, stride: int = 1, relu: bool = True, mode: str = 'original'):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()
        
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        
        # Primary conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        
        # Cheap operation
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        
        # DFC attention (for GhostV2)
        if mode == 'attn':
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2),
                          groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0),
                          groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'original':
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        else:
            # GhostV2 with DFC attention
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            out = out[:, :self.oup, :, :]
            return out * F.interpolate(self.gate_fn(res), size=out.shape[-2:], 
                                       mode='nearest')


# ===================================================================================
# TESTING
# ===================================================================================

if __name__ == "__main__":
    # Test Ghost Module
    x = torch.randn(2, 64, 32, 32)
    
    ghost = GhostModule(64, 128)
    out = ghost(x)
    print(f"GhostModule: {x.shape} -> {out.shape}")
    
    ghost_bn = GhostBottleneck(64, 128, 64, stride=2)
    out = ghost_bn(x)
    print(f"GhostBottleneck: {x.shape} -> {out.shape}")
    
    ghost_v2 = GhostModuleV2(64, 128, mode='attn')
    out = ghost_v2(x)
    print(f"GhostModuleV2: {x.shape} -> {out.shape}")