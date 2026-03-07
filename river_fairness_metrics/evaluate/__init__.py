"""Model evaluation.

This module provides utilities to evaluate an online model. The goal is to reproduce a real-world
scenario with high fidelity. The core function of this module is `progressive_val_score`, which
allows to evaluate a model via progressive validation.

"""
from __future__ import annotations

from .evaluation import iter_progressive_val_score, progressive_val_score

__all__ = [
    "iter_progressive_val_score",
    "progressive_val_score"
]