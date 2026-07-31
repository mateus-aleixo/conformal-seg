"""Per-image mask metrics. Everything takes prob/binary numpy arrays, HW."""

from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = (pred | target).sum()
    return float((pred & target).sum() / union) if union else 1.0


def pixel_precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    precision = tp / pred.sum() if pred.sum() else 1.0
    recall = tp / target.sum() if target.sum() else 1.0
    return float(precision), float(recall)


def fnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of true defect pixels the mask missed — the conformal loss.

    Defect-free targets contribute 0 (nothing to miss), keeping losses in [0,1]
    as conformal risk control requires.
    """
    target = target.astype(bool)
    total = target.sum()
    if total == 0:
        return 0.0
    missed = (target & ~pred.astype(bool)).sum()
    return float(missed / total)
