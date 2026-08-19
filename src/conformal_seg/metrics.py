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


def instance_fnr(pred: np.ndarray, target: np.ndarray, min_overlap: float = 0.10) -> float:
    """Fraction of defect *instances* the mask missed.

    The pixel-level FNR in `fnr` asks "what fraction of defect pixels did we
    keep?". On a hairline defect that is a brutal question: a one-pixel-wide
    thread has almost no interior, so keeping 90% of its pixels means keeping
    almost every marginal pixel, which forces the threshold below the model's
    noise floor and floods the frame. That is measured in docs/results.md.

    An inspector asks a different question: was the defect *found*. This loss
    labels connected components of the ground truth and counts an instance as
    found when the predicted mask covers at least `min_overlap` of it. Missing a
    thread entirely still costs; clipping its ends does not.

    Like `fnr`, this is nondecreasing in the threshold (a higher threshold can
    only shrink the mask, so an instance that was found can stop being found but
    never start), which is what conformal risk control needs. Defect-free targets
    contribute 0: there is nothing to find.
    """
    import cv2

    target = target.astype(np.uint8)
    if target.sum() == 0:
        return 0.0
    n_labels, labels = cv2.connectedComponents(target, connectivity=8)
    pred = pred.astype(bool)

    missed = 0
    instances = 0
    for label in range(1, n_labels):  # 0 is background
        component = labels == label
        size = int(component.sum())
        if size == 0:
            continue
        instances += 1
        if (pred & component).sum() < min_overlap * size:
            missed += 1
    return float(missed / instances) if instances else 0.0
