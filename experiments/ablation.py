"""Compatibility wrapper for 05_ablation.py."""

from ._compat import load_numeric_module

_module = load_numeric_module("05_ablation.py")

parse_args = _module.parse_args
quick_train = _module.quick_train