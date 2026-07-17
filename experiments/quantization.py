"""Compatibility wrapper for 04_quantization.py."""

from ._compat import load_numeric_module

_module = load_numeric_module("04_quantization.py")

run_quantization_experiment = _module.run_quantization_experiment
compare_quantization_methods = _module.compare_quantization_methods
parse_args = _module.parse_args