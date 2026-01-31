"""
===================================================================================
Learning Rate Scheduler Utilities
===================================================================================
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        warmup_lr: Starting learning rate for warmup
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 warmup_lr: float = 1e-6,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1):
        
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr + alpha * (base_lr - self.warmup_lr) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


class WarmupMultiStepScheduler(_LRScheduler):
    """
    Multi-step learning rate scheduler with linear warmup.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 milestones: List[int],
                 warmup_epochs: int = 5,
                 warmup_lr: float = 1e-6,
                 gamma: float = 0.1,
                 last_epoch: int = -1):
        
        self.milestones = sorted(milestones)
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.gamma = gamma
        
        super(WarmupMultiStepScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr + alpha * (base_lr - self.warmup_lr) 
                    for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 warmup_lr: float = 1e-6,
                 last_epoch: int = -1):
        
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr + alpha * (base_lr - self.warmup_lr) 
                    for base_lr in self.base_lrs]
        return self.base_lrs


def create_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_type: str,
                     total_epochs: int,
                     warmup_epochs: int = 5,
                     warmup_lr: float = 1e-6,
                     min_lr: float = 1e-6,
                     milestones: Optional[List[int]] = None,
                     gamma: float = 0.1,
                     step_size: int = 30) -> _LRScheduler:
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        total_epochs: Total training epochs
        warmup_epochs: Warmup epochs
        warmup_lr: Starting warmup learning rate
        min_lr: Minimum learning rate
        milestones: Milestones for multi-step scheduler
        gamma: Decay factor
        step_size: Step size for StepLR
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine' or scheduler_type == 'cosine_annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer, warmup_epochs, total_epochs, warmup_lr, min_lr
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_type == 'multistep':
        if milestones is None:
            milestones = [60, 120, 160]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    
    elif scheduler_type == 'warmup_multistep':
        if milestones is None:
            milestones = [60, 120, 160]
        return WarmupMultiStepScheduler(
            optimizer, milestones, warmup_epochs, warmup_lr, gamma
        )
    
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
    
    elif scheduler_type == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_epochs, pct_start=0.1
        )
    
    elif scheduler_type == 'none' or scheduler_type == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]['lr']


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    total_epochs = 200
    warmup_epochs = 5
    
    schedulers = {
        'cosine': create_scheduler(optimizer, 'cosine', total_epochs),
        'warmup_cosine': create_scheduler(torch.optim.SGD(model.parameters(), lr=0.1), 
                                          'warmup_cosine', total_epochs, warmup_epochs),
        'multistep': create_scheduler(torch.optim.SGD(model.parameters(), lr=0.1),
                                       'multistep', total_epochs),
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        for epoch in range(total_epochs):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        ax.plot(range(total_epochs), lrs, label=name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
