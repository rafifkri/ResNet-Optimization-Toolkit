"""Compatibility wrapper for 01_baseline.py."""

from ._compat import load_numeric_module

_module = load_numeric_module("01_baseline.py")

train_baseline = _module.train_baseline
parse_args = _module.parse_args