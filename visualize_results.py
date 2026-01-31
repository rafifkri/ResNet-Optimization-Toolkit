"""
Comprehensive Visualization for All Experiments
Generate charts for academic paper
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

def plot_accuracy_comparison():
    """Bar chart comparing all model accuracies"""
    models = ['Baseline\n(GhostResNet-18)', 'Pruned\n(50% sparsity)', 
              'Distilled\n(Student)', 'Quantized\n(INT8)']
    accuracies = [95.23, 93.00, 93.74, 95.22]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Model Accuracy Comparison on CIFAR-10', fontsize=16, fontweight='bold')
    ax.set_ylim(90, 97)
    ax.axhline(y=95.23, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    plt.tight_layout()
    plt.savefig('weights/visualization/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: accuracy_comparison.png")
    return fig

def plot_params_comparison():
    """Bar chart comparing model parameters"""
    models = ['Baseline', 'Pruned', 'Distilled', 'Quantized']
    params = [5.77, 5.77, 1.46, 5.77]  # in millions
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, p in zip(bars, params):
        height = bar.get_height()
        ax.annotate(f'{p:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Parameters (Millions)', fontsize=14)
    ax.set_title('Model Size Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 7)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/params_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: params_comparison.png")
    return fig

def plot_accuracy_vs_params():
    """Scatter plot: Accuracy vs Parameters (Pareto frontier)"""
    models = ['Baseline', 'Pruned', 'Distilled', 'Quantized']
    params = [5.77, 5.77, 1.46, 5.77]
    accuracies = [95.23, 93.00, 93.74, 95.22]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model, p, acc, c in zip(models, params, accuracies, colors):
        ax.scatter(p, acc, s=300, c=c, edgecolors='black', linewidth=2, label=model, zorder=5)
        ax.annotate(model, (p, acc), xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Parameters (Millions)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy vs Model Size Trade-off', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, 7)
    ax.set_ylim(91, 96)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/accuracy_vs_params.png', dpi=300, bbox_inches='tight')
    print("Saved: accuracy_vs_params.png")
    return fig

def plot_inference_speed():
    """Bar chart comparing inference throughput"""
    batch_sizes = [1, 8, 16, 32]
    cpu_throughput = [9.7, 28.3, 31.5, 45.7]
    gpu_throughput = [16.6, 155.8, 311.4, 616.4]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cpu_throughput, width, label='CPU', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, gpu_throughput, width, label='GPU (RTX 3050)', color='#2ecc71', edgecolor='black')
    
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=14)
    ax.set_title('Inference Throughput Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/inference_speed.png', dpi=300, bbox_inches='tight')
    print("Saved: inference_speed.png")
    return fig

def plot_latency_comparison():
    """Line plot showing latency across batch sizes"""
    batch_sizes = [1, 8, 16, 32]
    cpu_latency = [102.93, 282.50, 507.47, 699.48]
    gpu_latency = [60.07, 51.36, 51.38, 51.91]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, cpu_latency, 'o-', color='#3498db', linewidth=2, markersize=10, label='CPU')
    ax.plot(batch_sizes, gpu_latency, 's-', color='#2ecc71', linewidth=2, markersize=10, label='GPU (RTX 3050)')
    
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title('Inference Latency vs Batch Size', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: latency_comparison.png")
    return fig

def plot_optimization_summary():
    """Summary table as figure"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    data = [
        ['Baseline (GhostResNet-18)', '95.23%', '5.77M', '287.61M', '1.0x', '-'],
        ['Pruned (50% sparsity)', '93.00%', '5.77M', '~144M', '~2.0x', '-2.23%'],
        ['Distilled (Student)', '93.74%', '1.46M', '~72M', '~4.0x', '-1.49%'],
        ['Quantized (INT8)', '95.22%', '5.77M', '287.61M', '1.0x', '-0.01%'],
    ]
    
    columns = ['Model', 'Accuracy', 'Params', 'FLOPs', 'Compression', 'Acc Drop']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center',
                     colColours=['#3498db']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Model Optimization Summary', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/optimization_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: optimization_summary.png")
    return fig

def plot_compression_radar():
    """Radar chart showing different optimization aspects"""
    categories = ['Accuracy', 'Size\nReduction', 'Speed\nImprovement', 'Deployment\nReady', 'Training\nSimplicity']
    
    # Normalize scores (0-100)
    baseline = [95, 50, 50, 70, 100]
    pruned = [93, 60, 65, 60, 70]
    distilled = [94, 95, 75, 80, 60]
    quantized = [95, 55, 85, 95, 90]
    
    # Close the radar chart
    baseline += baseline[:1]
    pruned += pruned[:1]
    distilled += distilled[:1]
    quantized += quantized[:1]
    categories += categories[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, baseline, 'o-', linewidth=2, label='Baseline', color='#2ecc71')
    ax.fill(angles, baseline, alpha=0.1, color='#2ecc71')
    ax.plot(angles, pruned, 's-', linewidth=2, label='Pruned', color='#e74c3c')
    ax.fill(angles, pruned, alpha=0.1, color='#e74c3c')
    ax.plot(angles, distilled, '^-', linewidth=2, label='Distilled', color='#3498db')
    ax.fill(angles, distilled, alpha=0.1, color='#3498db')
    ax.plot(angles, quantized, 'd-', linewidth=2, label='Quantized', color='#9b59b6')
    ax.fill(angles, quantized, alpha=0.1, color='#9b59b6')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Optimization Methods Comparison', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('weights/visualization/compression_radar.png', dpi=300, bbox_inches='tight')
    print("Saved: compression_radar.png")
    return fig

def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS FOR EXPERIMENTS")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('weights/visualization', exist_ok=True)
    
    # Generate all plots
    print("\n1. Accuracy Comparison...")
    plot_accuracy_comparison()
    
    print("\n2. Parameters Comparison...")
    plot_params_comparison()
    
    print("\n3. Accuracy vs Parameters...")
    plot_accuracy_vs_params()
    
    print("\n4. Inference Speed...")
    plot_inference_speed()
    
    print("\n5. Latency Comparison...")
    plot_latency_comparison()
    
    print("\n6. Optimization Summary Table...")
    plot_optimization_summary()
    
    print("\n7. Compression Radar Chart...")
    plot_compression_radar()
    
    plt.close('all')
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS SAVED TO: weights/visualization/")
    print("=" * 60)
    
    # List generated files
    viz_dir = Path('weights/visualization')
    print("\nGenerated files:")
    for f in viz_dir.glob('*.png'):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
