"""Compatibility wrapper for 02_pruning.py."""

from ._compat import load_numeric_module

_module = load_numeric_module("02_pruning.py")

run_pruning_experiment = _module.run_pruning_experiment
parse_args = _module.parse_args