import numpy as np

from conformal_seg.conformal import calibrate, held_out_report, losses_at
from conformal_seg.metrics import fnr, iou, pixel_precision_recall


def test_metrics_basics():
    pred = np.zeros((8, 8), dtype=bool)
    target = np.zeros((8, 8), dtype=bool)
    target[2:4, 2:4] = True
    assert fnr(pred, target) == 1.0        # predicted nothing, missed all
    assert fnr(target, target) == 0.0      # perfect mask misses nothing
    assert fnr(pred, np.zeros((8, 8))) == 0.0  # nothing to miss
    assert iou(target, target) == 1.0
    p, r = pixel_precision_recall(target, target)
    assert p == r == 1.0


def _toy(n=40, size=32, seed=0):
    """Prob maps that are informative but imperfect, like a real model."""
    rng = np.random.default_rng(seed)
    probs, masks = [], []
    for _ in range(n):
        mask = np.zeros((size, size), dtype=bool)
        r0, c0 = rng.integers(4, size - 12, 2)
        mask[r0:r0 + 8, c0:c0 + 8] = True
        prob = rng.uniform(0, 0.35, (size, size))
        prob[mask] = rng.uniform(0.45, 1.0, mask.sum())
        probs.append(prob)
        masks.append(mask)
    return probs, masks


def test_fnr_monotone_in_threshold():
    probs, masks = _toy()
    l_low = losses_at(probs, masks, 0.2).mean()
    l_mid = losses_at(probs, masks, 0.6).mean()
    l_high = losses_at(probs, masks, 0.95).mean()
    assert l_low <= l_mid <= l_high


def test_calibrate_meets_alpha_on_held_out():
    cal_probs, cal_masks = _toy(n=60, seed=1)
    cal = calibrate(cal_probs, cal_masks, alpha=0.10)
    assert 0.0 <= cal.threshold <= 1.0
    # corrected risk at the chosen threshold satisfied alpha on calibration
    at_thr = [r for t, r in cal.risk_curve if abs(t - cal.threshold) < 1e-9]
    assert at_thr and at_thr[0] <= 0.10

    test_probs, test_masks = _toy(n=60, seed=2)
    report = held_out_report(test_probs, test_masks, cal)
    # E[FNR] <= alpha; allow sampling slack on a single held-out draw
    assert report["held_out_fnr_mean"] <= 0.10 + 0.05


def test_calibrate_falls_back_to_zero_threshold():
    """Uninformative probs: only t=0 (mask everything) can honour alpha."""
    rng = np.random.default_rng(3)
    probs = [rng.uniform(0, 0.05, (16, 16)) for _ in range(30)]
    masks = []
    for _ in range(30):
        m = np.zeros((16, 16), dtype=bool)
        m[4:8, 4:8] = True
        masks.append(m)
    cal = calibrate(probs, masks, alpha=0.05)
    assert cal.threshold == 0.0
    assert losses_at(probs, masks, cal.threshold).mean() == 0.0
