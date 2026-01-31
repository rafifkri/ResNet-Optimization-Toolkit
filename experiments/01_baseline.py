"""
===================================================================================
Experiment 01: Baseline Training
===================================================================================
Train baseline ResNet and Ghost-ResNet models with comprehensive logging,
evaluation, and checkpointing for research comparison.
===================================================================================
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig, get_cifar10_config, ModelType
from models import create_model, get_model_info
from utils import (
    get_loaders, accuracy, AverageMeter, setup_seed,
    ExperimentLogger, CheckpointManager, EarlyStopping,
    create_scheduler, get_lr, evaluate_model, get_model_complexity,
    mixup_data, mixup_criterion, cutmix_data, LabelSmoothingCrossEntropy
)


# ===================================================================================
# TRAINING FUNCTIONS
# ===================================================================================

def train_one_epoch(model: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    config: ExperimentConfig,
                    scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Experiment configuration
        scaler: GradScaler for mixed precision
    
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply Mixup or CutMix
        use_mixup = config.data.use_mixup and torch.rand(1).item() < 0.5
        use_cutmix = config.data.use_cutmix and torch.rand(1).item() < config.data.cutmix_prob
        
        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, config.data.cutmix_alpha)
            mixed = True
        elif use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.data.mixup_alpha)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if config.train.use_amp and scaler is not None:
            with autocast():
                outputs = model(inputs)
                if mixed:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                if config.train.grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                else:
                    nn.utils.clip_grad_value_(model.parameters(), config.train.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if mixed:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            if config.train.grad_clip > 0:
                if config.train.grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                else:
                    nn.utils.clip_grad_value_(model.parameters(), config.train.grad_clip)
            
            optimizer.step()
        
        # Calculate accuracy
        if not mixed:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
        else:
            acc1 = lam * accuracy(outputs, targets_a, topk=(1,))[0] + \
                   (1 - lam) * accuracy(outputs, targets_b, topk=(1,))[0]
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%',
            'LR': f'{get_lr(optimizer):.6f}'
        })
    
    return losses.avg, top1.avg


@torch.no_grad()
def validate(model: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
    
    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1, inputs.size(0))
    
    return losses.avg, top1.avg


# ===================================================================================
# MAIN TRAINING LOOP
# ===================================================================================

def train_baseline(config: ExperimentConfig, 
                   model_name: str = "ghost_resnet18") -> Dict:
    """
    Main training function.
    
    Args:
        config: Experiment configuration
        model_name: Name of model to train
    
    Returns:
        Dictionary with training results
    """
    # Setup
    setup_seed(config.train.seed, config.train.deterministic)
    device = config.get_device()
    config.setup_dirs()
    
    # Initialize logger (TensorBoard disabled due to NumPy 2.x incompatibility)
    logger = ExperimentLogger(
        experiment_name=config.train.experiment_name,
        log_dir=config.logs_dir,
        use_tensorboard=False,  # Disabled - NumPy 2.x + TensorFlow conflict
        use_wandb=config.train.use_wandb,
        wandb_project=config.train.wandb_project
    )
    
    logger.info(f"Starting training: {config.train.experiment_name}")
    logger.info(f"Device: {device}")
    logger.log_config(config)
    
    # Data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=config.data.batch_size,
        dataset=config.data.dataset.value,
        val_split=config.data.val_split,
        num_workers=config.data.num_workers
    )
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader) if val_loader else 0}, Test: {len(test_loader)}")
    
    # Create model
    logger.info(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        num_classes=config.model.num_classes,
        attention_type=config.model.attention_type if config.model.use_attention else "none",
        attention_reduction=config.model.attention_reduction,
        width_multiplier=config.model.width_multiplier,
        dropout_rate=config.model.dropout_rate,
        drop_path_rate=config.model.drop_path_rate
    )
    model = model.to(device)
    
    # Log model info
    model_info = get_model_info(model, input_size=(1, 3, config.data.image_size, config.data.image_size))
    logger.info(f"Parameters: {model_info['total_params_m']:.2f}M")
    if 'flops_m' in model_info:
        logger.info(f"FLOPs: {model_info['flops_m']:.2f}M")
    logger.log_model_summary(model, input_size=(1, 3, config.data.image_size, config.data.image_size))
    
    # Loss function
    if config.data.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.data.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    # Optimizer
    if config.train.optimizer.value == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
            nesterov=config.train.nesterov
        )
    elif config.train.optimizer.value == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.value == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay
        )
    
    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        config.train.scheduler.value,
        total_epochs=config.train.epochs,
        warmup_epochs=config.train.warmup_epochs,
        warmup_lr=config.train.warmup_lr,
        min_lr=config.train.min_lr,
        milestones=config.train.milestones,
        gamma=config.train.gamma
    )
    
    # Mixed precision
    scaler = GradScaler() if config.train.use_amp else None
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.weights_dir,
        experiment_name=config.train.experiment_name,
        max_checkpoints=5,
        monitor='val_acc',
        mode='max'
    )
    
    # Early stopping
    early_stopping = None
    if config.train.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.train.patience,
            min_delta=config.train.min_delta,
            mode='max'
        )
    
    # Resume from checkpoint
    start_epoch = 0
    if config.train.resume:
        logger.info(f"Resuming from {config.train.resume}")
        state = checkpoint_manager.load(config.train.resume)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': [], 'lr': []
    }
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0
    
    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, config, scaler
        )
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Test (optional, every N epochs)
        test_acc = None
        if epoch % 10 == 0 or epoch == config.train.epochs - 1:
            _, test_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = get_lr(optimizer)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if test_acc:
            history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        # Log epoch
        epoch_time = time.time() - epoch_start
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_acc=test_acc,
            lr=current_lr
        )
        logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        checkpoint_manager.save(
            state={
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            epoch=epoch,
            metrics={'val_acc': val_acc, 'val_loss': val_loss},
            is_best=is_best
        )
        
        # Early stopping
        if early_stopping and early_stopping(val_acc):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    logger.info("Final evaluation...")
    _, final_test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save results
    results = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_acc,
        'params_m': model_info['total_params_m'],
        'history': history,
    }
    
    logger.close()
    
    return results


# ===================================================================================
# MAIN
# ===================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--model', type=str, default='ghost_resnet18',
                        choices=['resnet18', 'resnet34', 'ghost_resnet18', 'ghost_resnet34'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--attention', type=str, default='coordinate',
                        choices=['none', 'se', 'cbam', 'eca', 'coordinate'])
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Get config
    config = get_cifar10_config()
    
    # Override with args
    config.train.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.model.attention_type = args.attention
    config.model.use_attention = args.attention != 'none'
    config.model.width_multiplier = args.width_mult
    config.train.experiment_name = args.experiment_name
    config.train.resume = args.resume
    config.train.use_amp = not args.no_amp
    
    if args.dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.model.num_classes = 100
    
    # Train
    results = train_baseline(config, model_name=args.model)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Val Acc: {results['best_val_acc']:.2f}%")
    print(f"Final Test Acc: {results['final_test_acc']:.2f}%")
    print("="*60)