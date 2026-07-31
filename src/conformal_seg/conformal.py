"""Conformal risk control for mask thresholds.

Loss per image at threshold t: FNR(prob >= t, truth) — the fraction of true
defect pixels the mask misses. FNR is nondecreasing in t (higher threshold,
smaller mask, more missed pixels), so the CRC condition is monotone and we take
the LARGEST threshold whose corrected risk meets alpha — the tightest mask that
still carries the guarantee:

    t̂ = max { t : (n/(n+1)) · R̂(t) + 1/(n+1) ≤ alpha }

Guarantee under exchangeability: E[FNR at t̂ on new images] ≤ alpha.
(Angelopoulos et al., 2022 — this is their segmentation example, implemented.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import fnr


@dataclass(frozen=True)
class Calibration:
    threshold: float
    alpha: float
    n: int
    risk_curve: tuple[tuple[float, float], ...]  # (threshold, corrected risk)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "n": self.n,
            "risk_curve": [list(p) for p in self.risk_curve],
        }


def losses_at(probs: list[np.ndarray], masks: list[np.ndarray], t: float) -> np.ndarray:
    return np.array([fnr(p >= t, m) for p, m in zip(probs, masks)], dtype=float)


def calibrate(
    probs: list[np.ndarray],
    masks: list[np.ndarray],
    alpha: float = 0.10,
    grid: np.ndarray | None = None,
) -> Calibration:
    """probs: per-image sigmoid maps in [0,1]; masks: binary ground truth."""
    if len(probs) != len(masks) or not probs:
        raise ValueError("probs and masks must be equal-length and non-empty")
    n = len(probs)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    curve: list[tuple[float, float]] = []
    best = 0.0  # t=0 predicts every pixel: FNR = 0 everywhere; always feasible
    for t in grid:
        risk = float(losses_at(probs, masks, float(t)).mean())
        corrected = (n / (n + 1)) * risk + 1.0 / (n + 1)
        curve.append((float(t), corrected))
        if corrected <= alpha:
            best = float(t)  # grid ascends: keep the largest feasible t
    return Calibration(threshold=best, alpha=alpha, n=n, risk_curve=tuple(curve))


def held_out_report(
    probs: list[np.ndarray], masks: list[np.ndarray], cal: Calibration
) -> dict:
    """The number that goes in docs/results.md: risk on data the threshold never saw."""
    losses = losses_at(probs, masks, cal.threshold)
    mask_sizes = [float((p >= cal.threshold).mean()) for p in probs]
    return {
        "alpha": cal.alpha,
        "threshold": cal.threshold,
        "held_out_fnr_mean": float(losses.mean()),
        "held_out_fnr_max": float(losses.max()),
        "mean_predicted_mask_fraction": float(np.mean(mask_sizes)),
        "n_test": len(probs),
    }
