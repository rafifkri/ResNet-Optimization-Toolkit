"""
===================================================================================
Advanced Data Augmentation Module
===================================================================================
Implements: Mixup, CutMix, Cutout, AutoAugment, RandAugment
===================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import random


# ===================================================================================
# CUTOUT
# ===================================================================================

class Cutout:
    """
    Randomly mask out one or more patches from an image.
    Reference: https://arxiv.org/abs/1708.04552
    """
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (C, H, W)
        Returns:
            Tensor with random square patches cut out
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


# ===================================================================================
# MIXUP
# ===================================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Mixup data augmentation
    Reference: https://arxiv.org/abs/1710.09412
    
    Args:
        x: Input tensor (batch)
        y: Labels
        alpha: Mixup interpolation coefficient
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===================================================================================
# CUTMIX
# ===================================================================================

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    CutMix data augmentation
    Reference: https://arxiv.org/abs/1905.04899
    
    Args:
        x: Input tensor (batch)
        y: Labels
        alpha: CutMix beta distribution parameter
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam


# ===================================================================================
# AUTOAUGMENT for CIFAR
# ===================================================================================

class CIFAR10Policy:
    """
    AutoAugment policies found on CIFAR-10.
    Reference: https://arxiv.org/abs/1805.09501
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            [("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
            [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
            [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
            [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)],
            [("ShearY", 0.2, 7), ("Posterize", 0.3, 7)],
            [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
            [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
            [("Equalize", 0.6, 5), ("Equalize", 0.5, 1)],
            [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)],
            [("Color", 0.7, 7), ("TranslateX", 0.5, 8)],
            [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
            [("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)],
            [("Brightness", 0.9, 6), ("Color", 0.2, 8)],
            [("Solarize", 0.5, 2), ("Invert", 0.0, 3)],
            [("Equalize", 0.2, 0), ("AutoContrast", 0.6, 0)],
            [("Equalize", 0.2, 8), ("Equalize", 0.6, 4)],
            [("Color", 0.9, 9), ("Equalize", 0.6, 6)],
            [("AutoContrast", 0.8, 4), ("Solarize", 0.2, 8)],
            [("Brightness", 0.1, 3), ("Color", 0.7, 0)],
            [("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
            [("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
            [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
            [("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)],
        ]
        self.fillcolor = fillcolor
    
    def __call__(self, img):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() < pr:
                img = apply_augment(img, name, level, self.fillcolor)
        return img


def apply_augment(img, name, level, fillcolor):
    """Apply a single augmentation operation"""
    # Mapping level (0-9) to actual magnitude
    ranges = {
        "ShearX": np.linspace(0, 0.3, 10),
        "ShearY": np.linspace(0, 0.3, 10),
        "TranslateX": np.linspace(0, 0.33, 10),
        "TranslateY": np.linspace(0, 0.33, 10),
        "Rotate": np.linspace(0, 30, 10),
        "Color": np.linspace(0, 0.9, 10),
        "Posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
        "Solarize": np.linspace(256, 0, 10),
        "Contrast": np.linspace(0, 0.9, 10),
        "Sharpness": np.linspace(0, 0.9, 10),
        "Brightness": np.linspace(0, 0.9, 10),
        "AutoContrast": [0] * 10,
        "Equalize": [0] * 10,
        "Invert": [0] * 10,
    }
    
    from PIL import ImageOps, ImageEnhance, ImageFilter
    
    if name == "ShearX":
        img = img.transform(img.size, Image.AFFINE, (1, ranges[name][level], 0, 0, 1, 0), fillcolor=fillcolor)
    elif name == "ShearY":
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, ranges[name][level], 1, 0), fillcolor=fillcolor)
    elif name == "TranslateX":
        img = img.transform(img.size, Image.AFFINE, (1, 0, ranges[name][level] * img.size[0], 0, 1, 0), fillcolor=fillcolor)
    elif name == "TranslateY":
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, ranges[name][level] * img.size[1]), fillcolor=fillcolor)
    elif name == "Rotate":
        img = img.rotate(ranges[name][level])
    elif name == "Color":
        img = ImageEnhance.Color(img).enhance(1 + ranges[name][level] * random.choice([-1, 1]))
    elif name == "Posterize":
        img = ImageOps.posterize(img, ranges[name][level])
    elif name == "Solarize":
        img = ImageOps.solarize(img, int(ranges[name][level]))
    elif name == "Contrast":
        img = ImageEnhance.Contrast(img).enhance(1 + ranges[name][level] * random.choice([-1, 1]))
    elif name == "Sharpness":
        img = ImageEnhance.Sharpness(img).enhance(1 + ranges[name][level] * random.choice([-1, 1]))
    elif name == "Brightness":
        img = ImageEnhance.Brightness(img).enhance(1 + ranges[name][level] * random.choice([-1, 1]))
    elif name == "AutoContrast":
        img = ImageOps.autocontrast(img)
    elif name == "Equalize":
        img = ImageOps.equalize(img)
    elif name == "Invert":
        img = ImageOps.invert(img)
    
    return img


# ===================================================================================
# RANDAUGMENT
# ===================================================================================

class RandAugment:
    """
    RandAugment: Practical automated data augmentation with a reduced search space.
    Reference: https://arxiv.org/abs/1909.13719
    """
    def __init__(self, n: int = 2, m: int = 9, fillcolor=(128, 128, 128)):
        """
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude of each transformation (0-30)
        """
        self.n = n
        self.m = m
        self.fillcolor = fillcolor
        self.augment_list = [
            "Identity", "AutoContrast", "Equalize", "Rotate",
            "Solarize", "Color", "Posterize", "Contrast",
            "Brightness", "Sharpness", "ShearX", "ShearY",
            "TranslateX", "TranslateY"
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            if op == "Identity":
                continue
            level = min(self.m, 9)  # Cap at 9 for our ranges
            img = apply_augment(img, op, level, self.fillcolor)
        return img


# ===================================================================================
# LABEL SMOOTHING LOSS
# ===================================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    Reference: https://arxiv.org/abs/1512.00567
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Convert to one-hot if needed
        if target.dim() == 1:
            target = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        target = target * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_pred = torch.log_softmax(pred, dim=-1)
        loss = torch.sum(-target * log_pred, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ===================================================================================
# AUGMENTATION FACTORY
# ===================================================================================

def get_train_transforms(config):
    """
    Build training transforms based on config
    
    Args:
        config: DataConfig object
    
    Returns:
        torchvision.transforms.Compose
    """
    transforms_list = []
    
    # Basic augmentations
    if config.image_size == 32:  # CIFAR
        transforms_list.extend([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ])
    else:  # ImageNet-style
        transforms_list.extend([
            T.RandomResizedCrop(config.image_size),
            T.RandomHorizontalFlip(),
        ])
    
    # AutoAugment
    if config.use_autoaugment:
        transforms_list.append(CIFAR10Policy())
    
    # RandAugment
    if config.use_randaugment:
        transforms_list.append(RandAugment(n=config.randaug_n, m=config.randaug_m))
    
    # ToTensor and Normalize
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(config.mean, config.std),
    ])
    
    # Cutout (applied after ToTensor)
    if config.use_cutout:
        transforms_list.append(Cutout(n_holes=config.cutout_n_holes, length=config.cutout_length))
    
    return T.Compose(transforms_list)


def get_test_transforms(config):
    """
    Build test/validation transforms based on config
    
    Args:
        config: DataConfig object
    
    Returns:
        torchvision.transforms.Compose
    """
    transforms_list = []
    
    if config.image_size != 32:  # Not CIFAR
        transforms_list.extend([
            T.Resize(int(config.image_size * 256 / 224)),
            T.CenterCrop(config.image_size),
        ])
    
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(config.mean, config.std),
    ])
    
    return T.Compose(transforms_list)
