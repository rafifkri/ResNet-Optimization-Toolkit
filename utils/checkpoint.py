"""
===================================================================================
Checkpoint Management Module
===================================================================================
"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union


class CheckpointManager:
    """
    Manages model checkpoints with best model tracking.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "./weights",
                 experiment_name: str = "experiment",
                 max_checkpoints: int = 5,
                 save_best_only: bool = False,
                 monitor: str = "val_acc",
                 mode: str = "max"):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this experiment
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when metric improves
            monitor: Metric to monitor for best model
            mode: 'max' or 'min' for the monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        self.checkpoint_files = []
    
    def _is_better(self, current: float) -> bool:
        """Check if current metric is better than best"""
        if self.mode == 'max':
            return current > self.best_value
        return current < self.best_value
    
    def save(self, 
             state: Dict[str, Any],
             epoch: int,
             metrics: Optional[Dict[str, float]] = None,
             is_best: Optional[bool] = None) -> str:
        """
        Save checkpoint.
        
        Args:
            state: State dict containing model, optimizer, etc.
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Override automatic best detection
        
        Returns:
            Path to saved checkpoint
        """
        # Determine if this is the best model
        if is_best is None and metrics is not None and self.monitor in metrics:
            current_value = metrics[self.monitor]
            is_best = self._is_better(current_value)
            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
        
        # Add metadata to state
        state['epoch'] = epoch
        state['best_value'] = self.best_value
        state['best_epoch'] = self.best_epoch
        if metrics:
            state['metrics'] = metrics
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(state, checkpoint_path)
            self.checkpoint_files.append(checkpoint_path)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"New best model saved! {self.monitor}: {self.best_value:.4f}")
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(state, latest_path)
        
        return str(self.checkpoint_dir / "best_model.pth" if is_best else latest_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        while len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    
    def load(self, 
             checkpoint_path: Optional[str] = None,
             load_best: bool = False,
             map_location: str = 'cpu') -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load the best model
            map_location: Device to map tensors to
        
        Returns:
            Loaded state dict
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / "latest.pth"
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        state = torch.load(path, map_location=map_location)
        
        # Restore tracking variables
        if 'best_value' in state:
            self.best_value = state['best_value']
        if 'best_epoch' in state:
            self.best_epoch = state['best_epoch']
        
        print(f"Loaded checkpoint from {path}")
        if 'epoch' in state:
            print(f"  Epoch: {state['epoch']}")
        if 'metrics' in state:
            print(f"  Metrics: {state['metrics']}")
        
        return state
    
    def load_model(self,
                   model: torch.nn.Module,
                   checkpoint_path: Optional[str] = None,
                   load_best: bool = True,
                   strict: bool = True) -> torch.nn.Module:
        """
        Load model weights from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Specific checkpoint path
            load_best: Load best model
            strict: Strict loading
        
        Returns:
            Model with loaded weights
        """
        state = self.load(checkpoint_path, load_best)
        
        if 'model' in state:
            state_dict = state['model']
        elif 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        
        # Handle DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=strict)
        return model
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best checkpoint"""
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'monitor': self.monitor,
            'mode': self.mode,
        }


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(self, 
                 patience: int = 20,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' or 'min' for the monitored metric
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: Stopping training")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping counter"""
        self.counter = 0
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.early_stop = False


def save_checkpoint(state: Dict[str, Any], 
                    filepath: str, 
                    is_best: bool = False,
                    best_path: Optional[str] = None):
    """
    Simple checkpoint saving function.
    
    Args:
        state: State dict to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
        best_path: Path to save best model
    """
    torch.save(state, filepath)
    
    if is_best and best_path:
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath: str, 
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    map_location: str = 'cpu') -> Dict[str, Any]:
    """
    Simple checkpoint loading function.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        map_location: Device mapping
    
    Returns:
        Loaded checkpoint dict
    """
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint


if __name__ == "__main__":
    # Test checkpoint manager
    import torch.nn as nn
    
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        experiment_name="test",
        max_checkpoints=3
    )
    
    # Simulate training
    for epoch in range(5):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        metrics = {'val_acc': 0.8 + epoch * 0.02}
        manager.save(state, epoch, metrics)
    
    # Load best
    loaded = manager.load(load_best=True)
    print(f"Best epoch: {loaded['epoch']}")
    
    # Cleanup test
    import shutil
    shutil.rmtree("./test_checkpoints", ignore_errors=True)
