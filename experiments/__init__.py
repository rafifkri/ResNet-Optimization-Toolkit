"""
experiments/__init__.py

This module provides easy access to all experiment scripts.
"""

# Rename files for proper module imports
# Note: Files are named with numbers for ordering but we provide clean names here

# Import guard - these will only work when running as scripts
try:
    from experiments import baseline as baseline
except ImportError:
    baseline = None

# Experiment module references
EXPERIMENTS = {
    'baseline': '01_baseline.py',
    'pruning': '02_pruning.py', 
    'distillation': '03_distillation.py',
    'quantization': '04_quantization.py',
    'ablation': '05_ablation.py'
}

__all__ = ['EXPERIMENTS', 'baseline']
