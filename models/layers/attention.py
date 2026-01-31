"""
===================================================================================
Attention Modules for ResNet Optimization
===================================================================================
Implements: SE (Squeeze-and-Excitation), CBAM, ECA, Coordinate Attention
===================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================================================================================
# SQUEEZE-AND-EXCITATION (SE) ATTENTION
# ===================================================================================

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Networks
    Reference: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ===================================================================================
# EFFICIENT CHANNEL ATTENTION (ECA)
# ===================================================================================

class ECAAttention(nn.Module):
    """
    Efficient Channel Attention
    Reference: https://arxiv.org/abs/1910.03151
    
    Uses 1D convolution instead of FC layers for efficiency
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECAAttention, self).__init__()
        # Adaptive kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ===================================================================================
# CBAM: Convolutional Block Attention Module
# ===================================================================================

class ChannelAttention(nn.Module):
    """Channel attention sub-module for CBAM"""
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module for CBAM"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAMAttention(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    Reference: https://arxiv.org/abs/1807.06521
    
    Combines channel and spatial attention
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super(CBAMAttention, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ===================================================================================
# COORDINATE ATTENTION
# ===================================================================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design
    Reference: https://arxiv.org/abs/2103.02907
    
    Encodes channel relationships and long-range dependencies with 
    precise positional information.
    """
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        
        # Encode position along height and width
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


# ===================================================================================
# TRIPLET ATTENTION
# ===================================================================================

class BasicConv(nn.Module):
    """Basic convolution block with batch norm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    """Z-Pool: Combines max and avg along channel dimension"""
    def forward(self, x):
        return torch.cat([
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1)
        ], dim=1)


class TripletAttention(nn.Module):
    """
    Triplet Attention
    Reference: https://arxiv.org/abs/2010.03045
    
    Captures cross-dimension interaction using rotation operation
    """
    def __init__(self, kernel_size: int = 7, no_spatial: bool = False):
        super(TripletAttention, self).__init__()
        self.kernel_size = kernel_size
        self.no_spatial = no_spatial
        
        self.z_pool = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=kernel_size//2, relu=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: H×W plane
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        y_perm1 = self.z_pool(x_perm1)
        y_perm1 = self.conv(y_perm1)
        y_perm1 = y_perm1.permute(0, 2, 1, 3).contiguous()
        
        # Branch 2: H×C plane
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        y_perm2 = self.z_pool(x_perm2)
        y_perm2 = self.conv(y_perm2)
        y_perm2 = y_perm2.permute(0, 3, 2, 1).contiguous()
        
        # Branch 3: W×C plane (spatial)
        if not self.no_spatial:
            y = self.z_pool(x)
            y = self.conv(y)
            weight = (y.sigmoid() + y_perm1.sigmoid() + y_perm2.sigmoid()) / 3
        else:
            weight = (y_perm1.sigmoid() + y_perm2.sigmoid()) / 2
        
        return x * weight


# ===================================================================================
# ATTENTION FACTORY
# ===================================================================================

def get_attention(attention_type: str, channels: int, reduction: int = 16) -> nn.Module:
    """
    Factory function to create attention modules
    
    Args:
        attention_type: Type of attention ('se', 'eca', 'cbam', 'coordinate', 'triplet')
        channels: Number of input/output channels
        reduction: Reduction ratio for attention
    
    Returns:
        Attention module
    """
    attention_type = attention_type.lower()
    
    if attention_type == "se":
        return SEAttention(channels, reduction)
    elif attention_type == "eca":
        return ECAAttention(channels)
    elif attention_type == "cbam":
        return CBAMAttention(channels, reduction)
    elif attention_type == "coordinate" or attention_type == "ca":
        return CoordinateAttention(channels, channels, reduction)
    elif attention_type == "triplet":
        return TripletAttention()
    elif attention_type == "none" or attention_type is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# ===================================================================================
# TESTING
# ===================================================================================

if __name__ == "__main__":
    # Test all attention modules
    batch_size = 2
    channels = 64
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print("Testing attention modules...")
    print(f"Input shape: {x.shape}")
    
    for att_type in ["se", "eca", "cbam", "coordinate", "triplet"]:
        attention = get_attention(att_type, channels)
        out = attention(x)
        params = sum(p.numel() for p in attention.parameters())
        print(f"{att_type.upper():12s} | Output: {out.shape} | Params: {params:,}")