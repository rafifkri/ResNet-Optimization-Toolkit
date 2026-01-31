"""
===================================================================================
Enhanced Data Loader Module
===================================================================================
Supports: CIFAR-10, CIFAR-100, ImageNet, Custom datasets
Features: Train/Val split, Multi-worker loading, Advanced augmentations
===================================================================================
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Import config and augmentations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import DataConfig, DatasetType
    from utils.augmentations import get_train_transforms, get_test_transforms
except ImportError:
    # Fallback for direct execution
    pass


# ===================================================================================
# DATASET CLASSES
# ===================================================================================

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images from a folder structure:
    data/
        train/
            class1/
            class2/
        test/
            class1/
            class2/
    """
    def __init__(self, root: str, train: bool = True, transform=None):
        from torchvision.datasets import ImageFolder
        
        split = "train" if train else "test"
        self.dataset = ImageFolder(os.path.join(root, split), transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# ===================================================================================
# SIMPLE LOADERS (backward compatibility)
# ===================================================================================

def get_loaders(batch_size=128, dataset="cifar10", val_split=0.1, num_workers=4):
    """
    Get train, validation, and test data loaders (simplified interface)
    
    Args:
        batch_size: Batch size
        dataset: Dataset name ('cifar10', 'cifar100')
        val_split: Validation split ratio
        num_workers: Number of data loading workers
    
    Returns:
        (train_loader, val_loader, test_loader) or (train_loader, test_loader) if val_split=0
    """
    # Dataset-specific normalization
    if dataset.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
    elif dataset.lower() == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets
    trainset = DatasetClass(root='./data', train=True, download=True, transform=transform_train)
    testset = DatasetClass(root='./data', train=False, download=True, transform=transform_test)
    
    # Split train into train/val
    if val_split > 0:
        num_train = len(trainset)
        num_val = int(num_train * val_split)
        num_train = num_train - num_val
        
        generator = torch.Generator().manual_seed(42)
        trainset, valset = random_split(trainset, [num_train, num_val], generator=generator)
        
        val_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    if val_loader:
        return train_loader, val_loader, test_loader
    return train_loader, test_loader


# ===================================================================================
# ADVANCED LOADERS (with config)
# ===================================================================================

def get_loaders_from_config(config) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get train, validation, and test data loaders from config
    
    Args:
        config: DataConfig object
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Build transforms
    train_transform = get_train_transforms(config)
    test_transform = get_test_transforms(config)
    
    # Get datasets
    if config.dataset == DatasetType.CIFAR10:
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=True, transform=test_transform
        )
    elif config.dataset == DatasetType.CIFAR100:
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=False, download=True, transform=test_transform
        )
    elif config.dataset == DatasetType.IMAGENET:
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(config.data_root, 'train'), transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(config.data_root, 'val'), transform=test_transform
        )
    else:
        train_dataset = CustomDataset(config.data_root, train=True, transform=train_transform)
        test_dataset = CustomDataset(config.data_root, train=False, transform=test_transform)
    
    # Split training set into train/val
    val_dataset = None
    if config.val_split > 0:
        num_train = len(train_dataset)
        num_val = int(num_train * config.val_split)
        num_train = num_train - num_val
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            train_dataset, [num_train, num_val], generator=generator
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    return train_loader, val_loader, test_loader


# ===================================================================================
# CALIBRATION DATA LOADER (for Quantization)
# ===================================================================================

def get_calibration_loader(batch_size: int = 64, num_batches: int = 100, dataset: str = "cifar10") -> DataLoader:
    """
    Get a calibration data loader for quantization
    
    Args:
        batch_size: Batch size
        num_batches: Number of batches for calibration
        dataset: Dataset name
    
    Returns:
        Calibration DataLoader
    """
    if dataset.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        DatasetClass = torchvision.datasets.CIFAR100
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    full_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform)
    
    # Take subset for calibration
    num_samples = min(num_batches * batch_size, len(full_dataset))
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    subset = Subset(full_dataset, indices)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


# ===================================================================================
# DATASET INFO
# ===================================================================================

def get_dataset_info(dataset: str = "cifar10") -> Dict[str, Any]:
    """
    Get dataset statistics and information
    
    Args:
        dataset: Dataset name
    
    Returns:
        Dictionary with dataset info
    """
    if dataset.lower() == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "image_size": 32,
            "train_samples": 50000,
            "test_samples": 10000,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset.lower() == "cifar100":
        return {
            "name": "CIFAR-100",
            "num_classes": 100,
            "image_size": 32,
            "train_samples": 50000,
            "test_samples": 10000,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        }
    else:
        return {"name": dataset, "num_classes": "unknown"}


# ===================================================================================
# MAIN (for testing)
# ===================================================================================

if __name__ == "__main__":
    print("Testing data loaders...")
    
    train_loader, val_loader, test_loader = get_loaders(batch_size=128, val_split=0.1)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")