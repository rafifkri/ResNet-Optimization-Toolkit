"""
===================================================================================
Stochastic Depth (Drop Path) Module
===================================================================================
Reference: Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
===================================================================================
"""

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This is the same as the DropConnect impl for EfficientNet.
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Work with different dimensional tensors
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    During training, randomly drops entire residual branches, 
    encouraging the network to learn more robust features.
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


class StochasticDepth(nn.Module):
    """
    Wrapper for stochastic depth with linear decay schedule.
    
    Usage:
        # Create with survival probability
        sd = StochasticDepth(survival_prob=0.8)
        
        # In forward:
        out = residual_block(x)
        out = sd(out, x)  # out = dropped_out + identity
    """
    def __init__(self, survival_prob: float = 1.0):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
        self.drop_prob = 1 - survival_prob
    
    def forward(self, residual: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.
        
        Args:
            residual: Output from residual branch
            identity: Input (skip connection)
        
        Returns:
            residual + identity with potential dropping
        """
        if self.training and self.drop_prob > 0:
            residual = drop_path(residual, self.drop_prob, True)
        return residual + identity


def get_drop_path_rate(block_idx: int, total_blocks: int, max_drop_rate: float = 0.2) -> float:
    """
    Calculate drop path rate for a specific block using linear decay.
    
    Args:
        block_idx: Index of current block (0-indexed)
        total_blocks: Total number of blocks in the network
        max_drop_rate: Maximum drop rate for the last block
    
    Returns:
        Drop path rate for this block
    """
    return max_drop_rate * block_idx / (total_blocks - 1)


if __name__ == "__main__":
    # Test drop path
    x = torch.randn(4, 64, 32, 32)
    
    dp = DropPath(drop_prob=0.2)
    dp.train()
    
    out = dp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Drop prob: {dp.drop_prob}")
