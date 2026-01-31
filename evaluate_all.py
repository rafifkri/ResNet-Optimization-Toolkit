"""
Comprehensive Evaluation Script
Evaluate all trained models on CIFAR-10 test set
"""

import torch
import torch.nn as nn
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.resnet_ghost import GhostResNet, ghost_resnet18, ghost_resnet_small
from utils.dataloader import get_loaders

def load_model_weights(model, weight_path):
    """Load weights with various checkpoint formats"""
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Filter profiling keys
    state_dict = {k: v for k, v in state_dict.items() 
                  if not k.endswith(('total_ops', 'total_params'))}
    
    model.load_state_dict(state_dict, strict=False)
    return model

def evaluate_model(model, test_loader, device, model_name="Model"):
    """Evaluate model and return metrics"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Per-class accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_acc = {class_names[i]: 100 * class_correct[i] / class_total[i] 
                 for i in range(10) if class_total[i] > 0}
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracy': class_acc
    }

def main():
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load test data
    print("Loading CIFAR-10 test set...")
    _, _, test_loader = get_loaders(batch_size=128, dataset="cifar10", num_workers=2)
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    results = {}
    
    # Define models to evaluate
    models_to_eval = [
        {
            'name': 'Baseline (GhostResNet-18 + CoordAtt)',
            'weight_path': 'weights/baseline.pth',
            'model_fn': lambda: ghost_resnet18(num_classes=10, attention_type='coordinate')
        },
        {
            'name': 'Pruned Model (50% sparsity)',
            'weight_path': 'weights/pruning_experiments/pruning_structured_0.5_20260130_105549/pruned_model.pth',
            'model_fn': lambda: ghost_resnet18(num_classes=10, attention_type='coordinate')
        },
        {
            'name': 'Distilled Student',
            'weight_path': 'weights/distillation_experiments/distillation_ghost_resnet18_ghost_resnet_small_20260130_141604/experiment/best_model.pth',
            'model_fn': lambda: ghost_resnet_small(num_classes=10, attention_type='coordinate')
        },
        {
            'name': 'Quantized Model (Dynamic INT8)',
            'weight_path': 'weights/quantization_experiments/quantization_dynamic_ghost_resnet18_20260131_073959/quantized_model_scripted.pt',
            'model_fn': lambda: ghost_resnet18(num_classes=10, attention_type='coordinate'),
            'is_quantized': True
        },
    ]
    
    print("-" * 70)
    print(f"{'Model':<45} {'Accuracy':<12} {'Params':<12}")
    print("-" * 70)
    
    for model_info in models_to_eval:
        name = model_info['name']
        weight_path = model_info['weight_path']
        
        if not os.path.exists(weight_path):
            print(f"{name:<45} {'NOT FOUND':<12}")
            continue
        
        try:
            is_quantized = model_info.get('is_quantized', False)
            
            if is_quantized:
                # For quantized model, load TorchScript
                model = torch.jit.load(weight_path, map_location='cpu')
                eval_device = 'cpu'  # Quantized models run on CPU
            else:
                model = model_info['model_fn']()
                model = load_model_weights(model, weight_path)
                eval_device = device
            
            # Count parameters
            if hasattr(model, 'parameters'):
                params = sum(p.numel() for p in model.parameters()) / 1e6
            else:
                params = 0
            
            # Evaluate
            if is_quantized:
                # Evaluate quantized model on CPU
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                metrics = {'accuracy': accuracy, 'correct': correct, 'total': total}
            else:
                metrics = evaluate_model(model, test_loader, eval_device, name)
            
            results[name] = {
                'accuracy': metrics['accuracy'],
                'params_m': params,
                'weight_path': weight_path
            }
            
            print(f"{name:<45} {metrics['accuracy']:.2f}%{'':<6} {params:.2f}M")
            
        except Exception as e:
            print(f"{name:<45} ERROR: {str(e)[:20]}")
            continue
    
    print("-" * 70)
    print()
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"🏆 Best Model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.2f}%")
    
    # Save results
    output_path = 'weights/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
