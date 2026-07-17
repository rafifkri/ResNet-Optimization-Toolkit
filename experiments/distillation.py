"""Compatibility wrapper for 03_distillation.py."""

from ._compat import load_numeric_module

_module = load_numeric_module("03_distillation.py")

run_distillation_experiment = _module.run_distillation_experiment
self_distillation = _module.self_distillation
parse_args = _module.parse_args