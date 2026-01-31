"""
===================================================================================
Logger Module - Comprehensive Logging for Experiments
===================================================================================
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import csv


class ExperimentLogger:
    """
    Comprehensive logger for tracking experiments.
    
    Features:
    - Console and file logging
    - TensorBoard integration
    - Metric tracking and CSV export
    - Configuration saving
    """
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = "./logs",
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 wandb_project: str = "resnet-optimization"):
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                # Suppress TensorFlow/TensorBoard import warnings for NumPy 2.x compatibility
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # Try to import SummaryWriter - may fail with NumPy 2.x + TensorFlow
                    from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.info("TensorBoard logging enabled")
            except (ImportError, AttributeError, TypeError, Exception) as e:
                self.writer = None
                self.warning(f"TensorBoard not available: {type(e).__name__}")
                self.warning("Training will continue without TensorBoard logging")
        
        # Wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"{experiment_name}_{self.timestamp}",
                    dir=str(self.log_dir)
                )
                self.info("Weights & Biases logging enabled")
            except Exception as e:
                self.warning(f"Wandb not available: {type(e).__name__}")
        
        # Metrics history
        self.metrics_history = []
        self.best_metrics = {}
        
    def _setup_logging(self):
        """Setup Python logging"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
    
    def log_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        config_path = self.log_dir / "config.json"
        
        # Convert dataclasses to dicts with recursion protection
        seen = set()
        def convert(obj, depth=0):
            if depth > 10:  # Prevent deep recursion
                return str(obj)
            
            obj_id = id(obj)
            if obj_id in seen:
                return f"<circular reference to {type(obj).__name__}>"
            
            if hasattr(obj, '__dict__') and not isinstance(obj, type):
                seen.add(obj_id)
                try:
                    return {k: convert(v, depth + 1) for k, v in obj.__dict__.items() 
                            if not k.startswith('_')}
                finally:
                    seen.discard(obj_id)
            elif hasattr(obj, 'value') and hasattr(obj, 'name'):  # Enum
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [convert(i, depth + 1) for i in obj]
            elif isinstance(obj, dict):
                return {str(k): convert(v, depth + 1) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        config_dict = convert(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.info(f"Configuration saved to {config_path}")
        
        # Log to wandb
        if self.wandb_run:
            import wandb
            wandb.config.update(config_dict)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all backends"""
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Console log
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} | {metric_str}")
        
        # TensorBoard
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        
        # Wandb
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)
        
        # History
        metrics['step'] = step
        self.metrics_history.append(metrics)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: Optional[float] = None, val_acc: Optional[float] = None,
                  test_acc: Optional[float] = None, lr: Optional[float] = None):
        """Log epoch metrics"""
        metrics = {
            'train/loss': train_loss,
            'train/acc': train_acc,
        }
        
        if val_loss is not None:
            metrics['val/loss'] = val_loss
        if val_acc is not None:
            metrics['val/acc'] = val_acc
        if test_acc is not None:
            metrics['test/acc'] = test_acc
        if lr is not None:
            metrics['lr'] = lr
        
        self.log_metrics(metrics, epoch)
        
        # Track best
        if val_acc is not None and val_acc > self.best_metrics.get('best_val_acc', 0):
            self.best_metrics['best_val_acc'] = val_acc
            self.best_metrics['best_epoch'] = epoch
    
    def log_model_summary(self, model, input_size: tuple = (1, 3, 32, 32)):
        """Log model summary"""
        try:
            from torchinfo import summary
            model_summary = summary(model, input_size=input_size, verbose=0)
            summary_str = str(model_summary)
            
            # Save to file with UTF-8 encoding
            with open(self.log_dir / "model_summary.txt", 'w', encoding='utf-8') as f:
                f.write(summary_str)
            
            self.info(f"Model summary saved")
        except ImportError:
            # Fallback
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.info(f"Model: {total_params:,} total params, {trainable:,} trainable")
        except Exception as e:
            self.warning(f"Could not save model summary: {e}")
    
    def save_results(self):
        """Save metrics history to CSV"""
        if not self.metrics_history:
            return
        
        csv_path = self.log_dir / "metrics.csv"
        
        # Get all keys
        all_keys = set()
        for m in self.metrics_history:
            all_keys.update(m.keys())
        all_keys = sorted(list(all_keys))
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        
        self.info(f"Metrics saved to {csv_path}")
        
        # Save best metrics
        if self.best_metrics:
            best_path = self.log_dir / "best_metrics.json"
            with open(best_path, 'w') as f:
                json.dump(self.best_metrics, f, indent=2)
    
    def log_image(self, tag: str, image, step: int):
        """Log image to TensorBoard"""
        if self.writer:
            self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag: str, figure, step: int):
        """Log matplotlib figure to TensorBoard"""
        if self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """Close all logging backends"""
        self.save_results()
        
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        self.info("Logging closed")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class ProgressMeter:
    """Display progress with multiple meters"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # Test logger
    logger = ExperimentLogger("test_experiment", use_tensorboard=False, use_wandb=False)
    
    logger.info("Test info message")
    logger.warning("Test warning")
    
    for epoch in range(3):
        logger.log_epoch(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.1,
            train_acc=0.8 + epoch * 0.05,
            val_loss=1.1 - epoch * 0.1,
            val_acc=0.78 + epoch * 0.05
        )
    
    logger.close()
