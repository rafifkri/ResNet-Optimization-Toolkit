"""
===================================================================================
Visualization Module - Training Curves, Confusion Matrix, t-SNE, etc.
===================================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ===================================================================================
# TRAINING VISUALIZATION
# ===================================================================================

def plot_training_curves(history: Dict[str, List[float]],
                         save_path: Optional[str] = None,
                         title: str = "Training Curves") -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    if 'test_acc' in history:
        ax.plot(epochs, history['test_acc'], 'g--', label='Test Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_lr_schedule(lr_history: List[float],
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates per epoch
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    epochs = range(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ===================================================================================
# CONFUSION MATRIX
# ===================================================================================

def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          save_path: Optional[str] = None,
                          title: str = "Confusion Matrix",
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (N x N)
        class_names: List of class names
        normalize: Normalize by row (true class)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=ax, square=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


# ===================================================================================
# t-SNE VISUALIZATION
# ===================================================================================

@torch.no_grad()
def extract_features(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device,
                     max_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model's penultimate layer.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device
        max_samples: Maximum number of samples
    
    Returns:
        features, labels
    """
    model.eval()
    model = model.to(device)
    
    features = []
    labels = []
    
    # Hook to get features before final layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook on avgpool layer
    if hasattr(model, 'avgpool'):
        hook = model.avgpool.register_forward_hook(get_activation('features'))
    else:
        # Try to find adaptive pool
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                hook = module.register_forward_hook(get_activation('features'))
                break
    
    total = 0
    for inputs, targets in dataloader:
        if total >= max_samples:
            break
        
        inputs = inputs.to(device)
        _ = model(inputs)
        
        feat = activation['features']
        feat = feat.view(feat.size(0), -1)
        
        features.append(feat.cpu().numpy())
        labels.append(targets.numpy())
        
        total += inputs.size(0)
    
    hook.remove()
    
    features = np.concatenate(features, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    
    return features, labels


def plot_tsne(features: np.ndarray,
              labels: np.ndarray,
              class_names: Optional[List[str]] = None,
              save_path: Optional[str] = None,
              title: str = "t-SNE Visualization",
              perplexity: int = 30) -> plt.Figure:
    """
    Plot t-SNE visualization of features.
    
    Args:
        features: Feature matrix (N x D)
        labels: Labels (N,)
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        perplexity: t-SNE perplexity
    
    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=name, alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    return fig


# ===================================================================================
# MODEL COMPARISON VISUALIZATION
# ===================================================================================

def plot_model_comparison(comparison: Dict[str, Dict[str, float]],
                          metrics: List[str] = None,
                          save_path: Optional[str] = None,
                          title: str = "Model Comparison") -> plt.Figure:
    """
    Plot bar chart comparing models.
    
    Args:
        comparison: Dictionary of model_name -> metrics dict
        metrics: Metrics to compare
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['total_m', 'flops_m', 'mean_ms', 'accuracy']
    
    model_names = list(comparison.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [comparison[m].get(metric, 0) for m in model_names]
        
        bars = ax.bar(range(n_models), values, color=colors)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig


def plot_pareto_front(comparison: Dict[str, Dict[str, float]],
                      x_metric: str = 'flops_m',
                      y_metric: str = 'accuracy',
                      save_path: Optional[str] = None,
                      title: str = "Accuracy vs FLOPs Trade-off") -> plt.Figure:
    """
    Plot Pareto front of accuracy vs efficiency.
    
    Args:
        comparison: Dictionary of model_name -> metrics dict
        x_metric: X-axis metric (e.g., 'flops_m', 'total_m', 'mean_ms')
        y_metric: Y-axis metric (e.g., 'accuracy')
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for name, metrics in comparison.items():
        x = metrics.get(x_metric, 0)
        y = metrics.get(y_metric, 0)
        
        ax.scatter(x, y, s=100, alpha=0.7)
        ax.annotate(name, (x, y), textcoords="offset points", 
                    xytext=(5, 5), ha='left', fontsize=9)
    
    ax.set_xlabel(x_metric, fontsize=12)
    ax.set_ylabel(y_metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto plot saved to {save_path}")
    
    return fig


# ===================================================================================
# PRUNING VISUALIZATION
# ===================================================================================

def plot_sparsity_analysis(sparsity_results: Dict[str, float],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot layer-wise sparsity analysis.
    
    Args:
        sparsity_results: Dictionary with 'layer_sparsity' dict
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    layer_sparsity = sparsity_results.get('layer_sparsity', {})
    
    if not layer_sparsity:
        return None
    
    # Sort by sparsity
    sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1], reverse=True)
    names = [x[0].replace('.weight', '') for x in sorted_layers]
    values = [x[1] for x in sorted_layers]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
    
    colors = plt.cm.RdYlGn_r(np.array(values) / 100)
    bars = ax.barh(range(len(names)), values, color=colors)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_title(f"Layer-wise Sparsity (Global: {sparsity_results.get('global_sparsity', 0):.1f}%)",
                 fontsize=14, fontweight='bold')
    ax.axvline(x=sparsity_results.get('global_sparsity', 0), color='red', 
               linestyle='--', label='Global Sparsity')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ===================================================================================
# ABLATION STUDY VISUALIZATION
# ===================================================================================

def plot_ablation_results(results: Dict[str, float],
                          baseline_name: str = "Full Model",
                          save_path: Optional[str] = None,
                          title: str = "Ablation Study Results") -> plt.Figure:
    """
    Plot ablation study results.
    
    Args:
        results: Dictionary of variant_name -> accuracy
        baseline_name: Name of the baseline model
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    values = list(results.values())
    
    baseline_value = results.get(baseline_name, max(values))
    colors = ['green' if v >= baseline_value else 'salmon' for v in values]
    
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.axvline(x=baseline_value, color='blue', linestyle='--', 
               linewidth=2, label=f'{baseline_name}: {baseline_value:.2f}%')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ===================================================================================
# UTILITY FUNCTIONS
# ===================================================================================

def save_all_figures(output_dir: str):
    """Save all open matplotlib figures"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.savefig(output_dir / f"figure_{i}.png", dpi=300, bbox_inches='tight')


def load_history_from_csv(csv_path: str) -> Dict[str, List[float]]:
    """Load training history from CSV file"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict('list')


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization module...")
    
    # Fake training history
    history = {
        'train_loss': [2.0 - i * 0.1 for i in range(20)],
        'val_loss': [2.1 - i * 0.09 for i in range(20)],
        'train_acc': [50 + i * 2 for i in range(20)],
        'val_acc': [48 + i * 1.9 for i in range(20)],
    }
    
    fig = plot_training_curves(history, title="Test Training Curves")
    plt.show()
    
    # Fake confusion matrix
    cm = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(cm, np.random.randint(80, 100, 10))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    fig = plot_confusion_matrix(cm, class_names, title="Test Confusion Matrix")
    plt.show()
