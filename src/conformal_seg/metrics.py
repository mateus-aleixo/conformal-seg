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


def flagged_fraction(pred: np.ndarray) -> float:
    """Fraction of pixels the mask flags. On a defect-free part this is pure
    false alarm: every flagged pixel is wrong by construction."""
    return float(pred.astype(bool).mean())


def image_flagged(pred: np.ndarray, min_area_frac: float = 1e-3) -> bool:
    """Would an inspection line escalate this part to a human?

    A pixel mask is not a decision. Real lines act on a part when the flagged
    region is big enough to be worth looking at, so a single stray pixel does
    not stop the belt. `min_area_frac` is that trigger, as a fraction of the
    image; 1e-3 is ~100 px at 320x320.
    """
    return bool(pred.astype(bool).mean() >= min_area_frac)
