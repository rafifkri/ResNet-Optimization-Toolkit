"""Compatibility helpers for numbered experiment modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_numeric_module(module_filename: str) -> ModuleType:
    """Load a module stored in a numbered Python file within this package."""
    package_dir = Path(__file__).parent
    module_path = package_dir / module_filename
    module_name = f"{__package__}.{module_path.stem}"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module