"""Tests for classification evaluation metrics."""

from __future__ import annotations

import numpy as np

from utils.evaluator import macro_f1_score


def test_macro_f1_perfect_multiclass():
    """Macro-F1 is 100% when every class is predicted correctly."""
    targets = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 0, 1, 2]
    assert macro_f1_score(targets, preds) == 100.0


def test_macro_f1_empty():
    """Empty inputs return 0%."""
    assert macro_f1_score([], []) == 0.0


def test_macro_f1_imbalanced():
    """Macro-F1 averages per-class F1 (not accuracy)."""
    targets = np.array([0, 0, 0, 1])
    preds = np.array([0, 0, 1, 0])
    # class 0: precision 2/3, recall 2/3 -> f1 = 2/3
    # class 1: precision 0, recall 0 -> f1 = 0
    expected = (2.0 / 3.0) * 50.0
    assert abs(macro_f1_score(targets, preds) - expected) < 1e-6
