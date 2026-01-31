"""
03_distillation.py - Comprehensive Knowledge Distillation Experiment

This script implements various knowledge distillation techniques:
1. Response-based KD (Hinton et al., 2015)
2. Feature-based KD (FitNets, Romero et al., 2015)
3. Attention Transfer (Zagoruyko & Komodakis, 2017)
4. Relational KD (Park et al., CVPR 2019)
5. Progressive Knowledge Distillation
6. Self-Distillation

Reference Papers:
- Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
- FitNets: Hints for Thin Deep Nets (Romero et al., ICLR 2015)
- Paying More Attention to Attention (Zagoruyko & Komodakis, ICLR 2017)
- Relational Knowledge Distillation (Park et al., CVPR 2019)
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_cifar10_config, TrainConfig
from models import create_model
from utils.dataloader import get_loaders_from_config
from utils.metrics import get_model_complexity, measure_latency, evaluate_model
from utils.logger import ExperimentLogger, AverageMeter
from utils.checkpoint import CheckpointManager, EarlyStopping
from utils.scheduler import create_scheduler as get_scheduler
from utils.augmentations import LabelSmoothingCrossEntropy, mixup_data, cutmix_data


# ============================================================================
# Knowledge Distillation Loss Functions
# ============================================================================

class DistillationLoss(nn.Module):
    """
    Standard Knowledge Distillation loss combining soft and hard targets.
    
    L = Î± * L_soft + (1 - Î±) * L_hard
    
    where:
    - L_soft = KL(softmax(student/T) || softmax(teacher/T)) * T^2
    - L_hard = CrossEntropy(student, labels)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.9, 
                 label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        if label_smoothing > 0:
            self.hard_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.hard_loss = nn.CrossEntropyLoss()
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            targets: Ground truth labels
            
        Returns:
            loss: Combined distillation loss
            loss_dict: Dictionary with individual loss components
        """
        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = self.hard_loss(student_logits, targets)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        loss_dict = {
            'total': loss.item(),
            'soft': soft_loss.item(),
            'hard': hard_loss.item()
        }
        
        return loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based Knowledge Distillation (FitNets).
    
    Matches intermediate feature representations between teacher and student.
    Uses adapter layers to match different feature dimensions.
    """
    
    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        super().__init__()
        
        # Create adapter layers to match dimensions
        self.adapters = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch != t_ch:
                adapter = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
                nn.init.kaiming_normal_(adapter.weight)
            else:
                adapter = nn.Identity()
            self.adapters.append(adapter)
    
    def forward(self, student_features: List[torch.Tensor], 
                teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute feature matching loss."""
        loss = 0
        for adapter, s_feat, t_feat in zip(self.adapters, student_features, teacher_features):
            # Adapt student features
            s_feat = adapter(s_feat)
            
            # Normalize features
            s_feat = F.normalize(s_feat, p=2, dim=1)
            t_feat = F.normalize(t_feat, p=2, dim=1)
            
            # MSE loss on features
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss / len(student_features)


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer loss.
    
    Transfers attention maps from teacher to student.
    Attention map = mean of squared activation maps across channels.
    """
    
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p
    
    def attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention map from feature maps."""
        # features: (B, C, H, W)
        # Return attention: (B, H*W)
        return F.normalize(features.pow(self.p).mean(1).view(features.size(0), -1), dim=1)
    
    def forward(self, student_features: List[torch.Tensor],
                teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention transfer loss."""
        loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_attn = self.attention_map(s_feat)
            t_attn = self.attention_map(t_feat)
            
            # Match attention maps
            loss += (s_attn - t_attn).pow(2).mean()
        
        return loss / len(student_features)


class RelationalKDLoss(nn.Module):
    """
    Relational Knowledge Distillation loss (RKD).
    
    Transfers relations between instances rather than individual representations.
    - Distance-wise: Preserves pairwise distances
    - Angle-wise: Preserves angular relations
    """
    
    def __init__(self, distance_weight: float = 25.0, angle_weight: float = 50.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def pdist(self, features: torch.Tensor, squared: bool = False) -> torch.Tensor:
        """Compute pairwise distances."""
        # features: (B, D)
        feat_square = features.pow(2).sum(1).unsqueeze(1)
        dist_square = feat_square + feat_square.t() - 2 * torch.mm(features, features.t())
        dist_square = F.relu(dist_square)  # Numerical stability
        
        if squared:
            return dist_square
        else:
            eps = 1e-12
            return torch.sqrt(dist_square + eps)
    
    def distance_loss(self, student_features: torch.Tensor, 
                      teacher_features: torch.Tensor) -> torch.Tensor:
        """Distance-wise distillation loss."""
        # Flatten features
        s_feat = student_features.view(student_features.size(0), -1)
        t_feat = teacher_features.view(teacher_features.size(0), -1)
        
        # Compute pairwise distances
        s_dist = self.pdist(s_feat, squared=False)
        t_dist = self.pdist(t_feat, squared=False)
        
        # Normalize by mean distance
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)
        
        return F.smooth_l1_loss(s_dist, t_dist)
    
    def angle_loss(self, student_features: torch.Tensor,
                   teacher_features: torch.Tensor) -> torch.Tensor:
        """Angle-wise distillation loss."""
        # Flatten features
        s_feat = student_features.view(student_features.size(0), -1)
        t_feat = teacher_features.view(teacher_features.size(0), -1)
        
        # Normalize
        s_feat = F.normalize(s_feat, p=2, dim=1)
        t_feat = F.normalize(t_feat, p=2, dim=1)
        
        # Compute angle matrices (cosine similarity)
        s_angle = torch.mm(s_feat, s_feat.t())
        t_angle = torch.mm(t_feat, t_feat.t())
        
        return F.smooth_l1_loss(s_angle, t_angle)
    
    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute RKD loss."""
        d_loss = self.distance_loss(student_features, teacher_features)
        a_loss = self.angle_loss(student_features, teacher_features)
        
        return self.distance_weight * d_loss + self.angle_weight * a_loss


class CombinedDistillationLoss(nn.Module):
    """
    Combined distillation loss with multiple components:
    - Response-based KD
    - Feature-based KD
    - Attention Transfer
    - Relational KD
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.9,
                 feature_weight: float = 0.1,
                 attention_weight: float = 0.1,
                 rkd_weight: float = 0.1,
                 student_channels: List[int] = None,
                 teacher_channels: List[int] = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        
        self.response_loss = DistillationLoss(temperature, alpha, label_smoothing)
        
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.rkd_weight = rkd_weight
        
        if student_channels and teacher_channels and feature_weight > 0:
            self.feature_loss = FeatureDistillationLoss(student_channels, teacher_channels)
        else:
            self.feature_loss = None
        
        if attention_weight > 0:
            self.attention_loss = AttentionTransferLoss()
        else:
            self.attention_loss = None
        
        if rkd_weight > 0:
            self.rkd_loss = RelationalKDLoss()
        else:
            self.rkd_loss = None
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor,
                student_features: List[torch.Tensor] = None,
                teacher_features: List[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Compute combined distillation loss."""
        
        # Response-based loss
        loss, loss_dict = self.response_loss(student_logits, teacher_logits, targets)
        
        # Feature-based loss
        if self.feature_loss and student_features and teacher_features:
            feat_loss = self.feature_loss(student_features, teacher_features)
            loss = loss + self.feature_weight * feat_loss
            loss_dict['feature'] = feat_loss.item()
        
        # Attention transfer loss
        if self.attention_loss and student_features and teacher_features:
            attn_loss = self.attention_loss(student_features, teacher_features)
            loss = loss + self.attention_weight * attn_loss
            loss_dict['attention'] = attn_loss.item()
        
        # RKD loss (using final features)
        if self.rkd_loss and student_features and teacher_features:
            rkd_loss = self.rkd_loss(student_features[-1], teacher_features[-1])
            loss = loss + self.rkd_weight * rkd_loss
            loss_dict['rkd'] = rkd_loss.item()
        
        loss_dict['total'] = loss.item()
        
        return loss, loss_dict


# ============================================================================
# Feature Extraction Wrapper
# ============================================================================

class FeatureExtractor(nn.Module):
    """Wrapper to extract intermediate features from a model."""
    
    def __init__(self, model: nn.Module, layer_names: List[str] = None):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        
        # Default layer names for ResNet
        if layer_names is None:
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        self.layer_names = layer_names
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture features."""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning logits and intermediate features."""
        self.features = {}
        logits = self.model(x)
        
        # Get features in order
        features = [self.features[name] for name in self.layer_names if name in self.features]
        
        return logits, features
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    use_amp: bool = True,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    extract_features: bool = False
) -> Dict:
    """Train student for one epoch using knowledge distillation."""
    
    student.train()
    teacher.eval()
    
    scaler = GradScaler(enabled=use_amp)
    
    loss_meter = AverageMeter('Loss')
    soft_loss_meter = AverageMeter('Soft')
    hard_loss_meter = AverageMeter('Hard')
    acc_meter = AverageMeter('Acc')
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup/cutmix
        if use_mixup and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # Get teacher outputs (no gradient)
            with torch.no_grad():
                if extract_features:
                    teacher_logits, teacher_features = teacher(inputs)
                else:
                    teacher_logits = teacher(inputs)
                    teacher_features = None
            
            # Get student outputs
            if extract_features:
                student_logits, student_features = student(inputs)
            else:
                student_logits = student(inputs)
                student_features = None
            
            # Compute loss
            if mixed:
                loss1, loss_dict1 = criterion(
                    student_logits, teacher_logits, targets_a,
                    student_features, teacher_features
                )
                loss2, loss_dict2 = criterion(
                    student_logits, teacher_logits, targets_b,
                    student_features, teacher_features
                )
                loss = lam * loss1 + (1 - lam) * loss2
                loss_dict = {k: lam * loss_dict1[k] + (1-lam) * loss_dict2[k] 
                            for k in loss_dict1}
            else:
                loss, loss_dict = criterion(
                    student_logits, teacher_logits, targets,
                    student_features, teacher_features
                )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        with torch.no_grad():
            pred = student_logits.argmax(dim=1)
            if mixed:
                acc = (lam * pred.eq(targets_a).float() + 
                       (1-lam) * pred.eq(targets_b).float()).mean().item() * 100
            else:
                acc = pred.eq(targets).float().mean().item() * 100
        
        loss_meter.update(loss_dict['total'], inputs.size(0))
        soft_loss_meter.update(loss_dict.get('soft', 0), inputs.size(0))
        hard_loss_meter.update(loss_dict.get('hard', 0), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.3f}',
            'soft': f'{soft_loss_meter.avg:.3f}',
            'acc': f'{acc_meter.avg:.1f}%'
        })
    
    if scheduler is not None:
        scheduler.step()
    
    return {
        'loss': loss_meter.avg,
        'soft_loss': soft_loss_meter.avg,
        'hard_loss': hard_loss_meter.avg,
        'accuracy': acc_meter.avg
    }


def validate(model: nn.Module, val_loader, criterion: nn.Module, 
             device: str) -> Dict:
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    top5_meter = AverageMeter('Top5')
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = F.cross_entropy(outputs, targets)
            
            # Top-1 accuracy
            pred = outputs.argmax(dim=1)
            acc = pred.eq(targets).float().mean().item() * 100
            
            # Top-5 accuracy
            _, pred5 = outputs.topk(5, 1, True, True)
            correct5 = pred5.eq(targets.view(-1, 1).expand_as(pred5))
            top5 = correct5.any(dim=1).float().mean().item() * 100
            
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            top5_meter.update(top5, inputs.size(0))
    
    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg,
        'top5_accuracy': top5_meter.avg
    }


# ============================================================================
# Progressive Knowledge Distillation
# ============================================================================

def progressive_distillation(
    teachers: List[nn.Module],
    student_configs: List[Dict],
    train_loader,
    val_loader,
    device: str,
    epochs_per_stage: int = 50,
    temperature_schedule: List[float] = None,
    logger: ExperimentLogger = None
) -> nn.Module:
    """
    Progressive Knowledge Distillation.
    
    Train increasingly smaller students, each learning from the previous model.
    
    Args:
        teachers: List of teacher models (from largest to smallest)
        student_configs: List of student model configurations
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        epochs_per_stage: Epochs per distillation stage
        temperature_schedule: Temperature schedule for each stage
        logger: Experiment logger
    
    Returns:
        Final distilled student model
    """
    if temperature_schedule is None:
        temperature_schedule = [4.0] * len(teachers)
    
    current_teacher = teachers[0]
    current_teacher = current_teacher.to(device)
    current_teacher.eval()
    
    for stage, (config, temp) in enumerate(zip(student_configs, temperature_schedule)):
        if logger:
            logger.info(f"\n{'='*50}")
            logger.info(f"Progressive KD Stage {stage + 1}/{len(student_configs)}")
            logger.info(f"{'='*50}")
            logger.info(f"Student config: {config}")
            logger.info(f"Temperature: {temp}")
        
        # Create student
        student = create_model(**config).to(device)
        
        # Setup training
        criterion = DistillationLoss(temperature=temp, alpha=0.9)
        optimizer = optim.AdamW(student.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_stage)
        
        best_acc = 0
        best_state = None
        
        for epoch in range(1, epochs_per_stage + 1):
            # Train
            train_metrics = train_one_epoch(
                student, current_teacher, train_loader,
                criterion, optimizer, scheduler, device, epoch,
                extract_features=False
            )
            
            # Validate
            val_metrics = validate(student, val_loader, criterion, device)
            
            if logger:
                logger.info(f"Stage {stage+1} Epoch {epoch}: "
                          f"Train Loss={train_metrics['loss']:.3f}, "
                          f"Val Acc={val_metrics['accuracy']:.2f}%")
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_state = copy.deepcopy(student.state_dict())
        
        # Load best and use as next teacher
        student.load_state_dict(best_state)
        current_teacher = student
        current_teacher.eval()
        
        if logger:
            logger.info(f"Stage {stage+1} complete. Best accuracy: {best_acc:.2f}%")
    
    return current_teacher


# ============================================================================
# Self-Distillation
# ============================================================================

def self_distillation(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    num_generations: int = 3,
    epochs_per_generation: int = 50,
    temperature: float = 4.0,
    logger: ExperimentLogger = None
) -> Tuple[nn.Module, Dict]:
    """
    Self-Distillation: Model learns from itself across generations.
    
    Also known as Born-Again Networks (BANs).
    
    Args:
        model: Base model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        num_generations: Number of self-distillation generations
        epochs_per_generation: Epochs per generation
        temperature: Distillation temperature
        logger: Experiment logger
    
    Returns:
        Final self-distilled model and history
    """
    history = {'generation': [], 'accuracy': []}
    
    current_model = model.to(device)
    
    for gen in range(num_generations):
        if logger:
            logger.info(f"\n{'='*50}")
            logger.info(f"Self-Distillation Generation {gen + 1}/{num_generations}")
            logger.info(f"{'='*50}")
        
        # Teacher is previous generation (frozen)
        teacher = copy.deepcopy(current_model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        
        # Student is fresh copy with same architecture
        student = copy.deepcopy(model)
        if gen > 0:
            # Initialize from previous generation
            student.load_state_dict(current_model.state_dict())
        student = student.to(device)
        
        # Setup training
        criterion = DistillationLoss(temperature=temperature, alpha=0.9)
        optimizer = optim.AdamW(student.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_generation)
        
        best_acc = 0
        best_state = None
        
        for epoch in range(1, epochs_per_generation + 1):
            # Train
            train_metrics = train_one_epoch(
                student, teacher, train_loader,
                criterion, optimizer, scheduler, device, epoch,
                extract_features=False
            )
            
            # Validate
            val_metrics = validate(student, val_loader, criterion, device)
            
            if logger and epoch % 10 == 0:
                logger.info(f"Gen {gen+1} Epoch {epoch}: "
                          f"Train Loss={train_metrics['loss']:.3f}, "
                          f"Val Acc={val_metrics['accuracy']:.2f}%")
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_state = copy.deepcopy(student.state_dict())
        
        # Update current model
        student.load_state_dict(best_state)
        current_model = student
        
        history['generation'].append(gen + 1)
        history['accuracy'].append(best_acc)
        
        if logger:
            logger.info(f"Generation {gen+1} complete. Best accuracy: {best_acc:.2f}%")
        
        # Clean up
        del teacher
        torch.cuda.empty_cache()
    
    return current_model, history


# ============================================================================
# Main Training Function
# ============================================================================

def run_distillation_experiment(args):
    """Run complete knowledge distillation experiment."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get config
    config = get_cifar10_config()
    if args.dataset == 'cifar100':
        config.data.dataset = 'cifar100'
        config.data.num_classes = 100
    
    # Setup directories
    experiment_name = f"distillation_{args.teacher}_{args.student}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = ExperimentLogger(experiment_name, str(save_dir), use_tensorboard=False)  # Disabled - NumPy 2.x compatibility
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Data loaders
    config.data.batch_size = args.batch_size
    train_loader, val_loader, test_loader = get_loaders_from_config(config.data)
    num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, 'classes') else 10
    
    # Create teacher model
    teacher = create_model(
        args.teacher,
        num_classes=num_classes,
        attention_type=args.teacher_attention,
        pretrained=False
    )
    
    # Load teacher weights
    if args.teacher_path:
        if os.path.exists(args.teacher_path):
            print(f"Loading teacher from {args.teacher_path}")
            checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out profiling keys (total_ops, total_params from thop/ptflops)
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.endswith(('total_ops', 'total_params'))}
            teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded {len(state_dict)} parameters from checkpoint")
        else:
            print(f"Warning: Teacher path {args.teacher_path} not found.")
    
    teacher = teacher.to(device)
    teacher.eval()
    
    # Create student model
    student = create_model(
        args.student,
        num_classes=num_classes,
        attention_type=args.student_attention,
        pretrained=False
    )
    student = student.to(device)
    
    # Model analysis
    logger.info("\n" + "="*50)
    logger.info("Model Analysis")
    logger.info("="*50)
    
    teacher_complexity = get_model_complexity(teacher)
    student_complexity = get_model_complexity(student)
    
    logger.info(f"Teacher ({args.teacher}):")
    logger.info(f"  Parameters: {teacher_complexity['params_m']:.2f}M")
    logger.info(f"  FLOPs: {teacher_complexity['flops_m']:.2f}M")
    
    logger.info(f"Student ({args.student}):")
    logger.info(f"  Parameters: {student_complexity['params_m']:.2f}M")
    logger.info(f"  FLOPs: {student_complexity['flops_m']:.2f}M")
    
    compression = teacher_complexity['params_m'] / max(student_complexity['params_m'], 1e-6)
    logger.info(f"\nCompression ratio: {compression:.2f}x")
    
    # Validate teacher
    teacher_val = validate(teacher, val_loader, nn.CrossEntropyLoss(), device)
    logger.info(f"\nTeacher validation accuracy: {teacher_val['accuracy']:.2f}%")
    
    # Setup for feature-based distillation
    extract_features = args.feature_weight > 0 or args.attention_weight > 0 or args.rkd_weight > 0
    
    if extract_features:
        # Wrap models for feature extraction
        teacher = FeatureExtractor(teacher)
        student = FeatureExtractor(student)
        
        # Get feature channel sizes (simplified - assumes ResNet structure)
        # In practice, you'd inspect the actual model architecture
        student_channels = [64, 128, 256, 512]  # Default ResNet18
        teacher_channels = [256, 512, 1024, 2048]  # Default ResNet50
        
        if 'ghost' in args.student or 'small' in args.student:
            student_channels = [32, 64, 128, 256]
    else:
        student_channels = None
        teacher_channels = None
    
    # Setup criterion
    criterion = CombinedDistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=args.feature_weight,
        attention_weight=args.attention_weight,
        rkd_weight=args.rkd_weight,
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        label_smoothing=args.label_smoothing
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        student.parameters() if not extract_features else student.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        str(save_dir),
        max_checkpoints=3,
        save_best_only=True
    )
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Knowledge Distillation Training")
    logger.info("="*50)
    
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            student, teacher, train_loader,
            criterion, optimizer, scheduler, device, epoch,
            use_amp=args.amp,
            use_mixup=args.mixup,
            mixup_alpha=args.mixup_alpha,
            extract_features=extract_features
        )
        
        # Validate
        student_model = student.model if extract_features else student
        val_metrics = validate(student_model, val_loader, nn.CrossEntropyLoss(), device)
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss={train_metrics['loss']:.3f}, "
                  f"Train Acc={train_metrics['accuracy']:.1f}%, "
                  f"Val Acc={val_metrics['accuracy']:.2f}%, "
                  f"LR={lr:.6f}")
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(lr)
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
        
        ckpt_manager.save(
            state={
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            epoch=epoch,
            metrics=val_metrics,
            is_best=is_best
        )
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation")
    logger.info("="*50)
    
    # Try to load best model, fallback to latest
    best_model_path = save_dir / 'best_model.pth'
    latest_model_path = save_dir / 'latest.pth'
    
    if best_model_path.exists():
        checkpoint_path = best_model_path
    elif latest_model_path.exists():
        checkpoint_path = latest_model_path
        logger.info("Best model not found, using latest checkpoint")
    else:
        logger.info("No checkpoint found, using current model state")
        checkpoint_path = None
    
    if checkpoint_path is not None:
        best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in best_checkpoint:
            student_model.load_state_dict(best_checkpoint['model_state_dict'])
        elif 'state_dict' in best_checkpoint:
            student_model.load_state_dict(best_checkpoint['state_dict'])
        else:
            # Maybe the checkpoint is just the state dict
            student_model.load_state_dict(best_checkpoint)
    
    test_metrics = validate(student_model, test_loader, nn.CrossEntropyLoss(), device)
    latency_stats = measure_latency(student_model, (1, 3, 32, 32), device=str(device))
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"Test Top-5 Accuracy: {test_metrics['top5_accuracy']:.2f}%")
    logger.info(f"Teacher Accuracy: {teacher_val['accuracy']:.2f}%")
    logger.info(f"Accuracy Gap: {teacher_val['accuracy'] - test_metrics['accuracy']:.2f}%")
    logger.info(f"Latency: {latency_stats['mean_ms']:.2f} Â± {latency_stats['std_ms']:.2f} ms")
    
    # Save results
    summary = {
        'teacher': {
            'model': args.teacher,
            'accuracy': teacher_val['accuracy'],
            'params_m': teacher_complexity['params_m'],
            'flops_m': teacher_complexity['flops_m']
        },
        'student': {
            'model': args.student,
            'accuracy': test_metrics['accuracy'],
            'top5_accuracy': test_metrics['top5_accuracy'],
            'params_m': student_complexity['params_m'],
            'flops_m': student_complexity['flops_m'],
            'latency_ms': latency_stats['mean_ms']
        },
        'distillation': {
            'temperature': args.temperature,
            'alpha': args.alpha,
            'feature_weight': args.feature_weight,
            'attention_weight': args.attention_weight,
            'rkd_weight': args.rkd_weight
        },
        'compression_ratio': compression,
        'accuracy_gap': teacher_val['accuracy'] - test_metrics['accuracy']
    }
    
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nResults saved to {save_dir}")
    logger.close()
    
    return student_model, summary


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Experiment')
    
    # Model
    parser.add_argument('--teacher', type=str, default='ghost_resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'ghost_resnet18', 'ghost_resnet34'],
                        help='Teacher model architecture')
    parser.add_argument('--teacher_attention', type=str, default='coordinate',
                        choices=['none', 'se', 'cbam', 'eca', 'coord', 'coordinate'])
    parser.add_argument('--teacher_path', type=str, default='weights/teacher.pth',
                        help='Path to pretrained teacher')
    
    parser.add_argument('--student', type=str, default='ghost_resnet_small',
                        choices=['resnet18', 'ghost_resnet18', 'ghost_resnet_small', 'ghost_resnet_tiny'],
                        help='Student model architecture')
    parser.add_argument('--student_attention', type=str, default='coordinate',
                        choices=['none', 'se', 'cbam', 'eca', 'coord', 'coordinate'])
    parser.add_argument('--width_multiplier', type=float, default=1.0,
                        help='Student width multiplier')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Distillation
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Weight for soft loss (1-alpha for hard loss)')
    parser.add_argument('--feature_weight', type=float, default=0.0,
                        help='Weight for feature distillation loss')
    parser.add_argument('--attention_weight', type=float, default=0.0,
                        help='Weight for attention transfer loss')
    parser.add_argument('--rkd_weight', type=float, default=0.0,
                        help='Weight for relational KD loss')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    
    # Augmentation
    parser.add_argument('--mixup', action='store_true', default=True)
    parser.add_argument('--no-mixup', action='store_false', dest='mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    
    # AMP
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    
    # Special modes
    parser.add_argument('--self_distill', action='store_true',
                        help='Run self-distillation (BANs)')
    parser.add_argument('--generations', type=int, default=3,
                        help='Number of generations for self-distillation')
    
    # Save
    parser.add_argument('--save_dir', type=str, default='weights/distillation_experiments')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.self_distill:
        # Self-distillation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = get_cifar10_config()
        train_loader, val_loader, _, num_classes = get_loaders_from_config(config.data)
        
        model = create_model(args.student, num_classes=num_classes)
        
        save_dir = Path(args.save_dir) / f"self_distill_{args.student}"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = ExperimentLogger("self_distillation", str(save_dir), use_tensorboard=False)  # Disabled - NumPy 2.x compatibility
        
        model, history = self_distillation(
            model, train_loader, val_loader, device,
            num_generations=args.generations,
            epochs_per_generation=args.epochs // args.generations,
            temperature=args.temperature,
            logger=logger
        )
        
        torch.save(model.state_dict(), save_dir / 'self_distilled_model.pth')
        logger.close()
    else:
        run_distillation_experiment(args)
