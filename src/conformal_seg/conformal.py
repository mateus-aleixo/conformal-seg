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

from .metrics import flagged_fraction, fnr, image_flagged, instance_fnr


@dataclass(frozen=True)
class Calibration:
    threshold: float
    alpha: float
    n: int
    risk_curve: tuple[tuple[float, float], ...]  # (threshold, corrected risk)
    # Which loss the threshold was fitted under. A threshold without this is
    # meaningless: the same number bounds a different quantity under each.
    loss: str = "pixel"

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "n": self.n,
            "loss": self.loss,
            "risk_curve": [list(p) for p in self.risk_curve],
        }


# The loss the guarantee is *about*. Both are per-image, in [0, 1], and
# nondecreasing in the threshold, which is all conformal risk control requires.
# They ask different questions, and on hairline defects the answers diverge
# sharply: see docs/results.md.
LOSSES = {
    "pixel": fnr,            # fraction of true defect PIXELS missed
    "instance": instance_fnr,  # fraction of defect INSTANCES missed
}


def losses_at(
    probs: list[np.ndarray],
    masks: list[np.ndarray],
    t: float,
    loss: str = "pixel",
) -> np.ndarray:
    fn = LOSSES[loss]
    return np.array(
        [fn(p >= t, m) for p, m in zip(probs, masks, strict=True)], dtype=float
    )


def calibrate(
    probs: list[np.ndarray],
    masks: list[np.ndarray],
    alpha: float = 0.10,
    grid: np.ndarray | None = None,
    loss: str = "pixel",
) -> Calibration:
    """probs: per-image sigmoid maps in [0,1]; masks: binary ground truth.

    `loss` selects what the guarantee is about: "pixel" bounds the fraction of
    defect pixels missed, "instance" the fraction of defect instances missed.
    """
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {sorted(LOSSES)}")
    if len(probs) != len(masks) or not probs:
        raise ValueError("probs and masks must be equal-length and non-empty")
    n = len(probs)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    curve: list[tuple[float, float]] = []
    best = 0.0  # t=0 predicts every pixel: FNR = 0 everywhere; always feasible
    for t in grid:
        risk = float(losses_at(probs, masks, float(t), loss).mean())
        corrected = (n / (n + 1)) * risk + 1.0 / (n + 1)
        curve.append((float(t), corrected))
        if corrected <= alpha:
            best = float(t)  # grid ascends: keep the largest feasible t
    return Calibration(
        threshold=best, alpha=alpha, n=n, loss=loss, risk_curve=tuple(curve)
    )


def held_out_report(
    probs: list[np.ndarray], masks: list[np.ndarray], cal: Calibration
) -> dict:
    """The number that goes in docs/results.md: risk on data the threshold never saw."""
    losses = losses_at(probs, masks, cal.threshold, cal.loss)
    mask_sizes = [float((p >= cal.threshold).mean()) for p in probs]
    return {
        "alpha": cal.alpha,
        "threshold": cal.threshold,
        "loss": cal.loss,
        "held_out_fnr_mean": float(losses.mean()),
        "held_out_fnr_max": float(losses.max()),
        "mean_predicted_mask_fraction": float(np.mean(mask_sizes)),
        "n_test": len(probs),
    }


def control_report(
    good_probs: list[np.ndarray],
    thresholds: dict[str, float],
    min_area_frac: float = 1e-3,
) -> dict:
    """False alarms on defect-free parts, at each named threshold.

    The conformal guarantee bounds what the mask MISSES on defective parts. It
    says nothing whatsoever about clean ones, and it buys its miss rate by
    lowering the threshold, which can only flag more. This is the other half of
    the decision the README claims to answer -- "can this part ship without a
    human looking at it?" -- measured on parts that are fine.

    Reported per threshold so the naive and conformal operating points can be
    compared on the same parts.
    """
    if not good_probs:
        return {"n_control": 0, "min_area_frac": min_area_frac, "at": {}}
    out: dict = {
        "n_control": len(good_probs),
        "min_area_frac": min_area_frac,
        "at": {},
    }
    for name, t in thresholds.items():
        preds = [p >= float(t) for p in good_probs]
        flagged = [flagged_fraction(q) for q in preds]
        alarms = [image_flagged(q, min_area_frac) for q in preds]
        out["at"][name] = {
            "threshold": float(t),
            "mean_flagged_pixel_fraction": float(np.mean(flagged)),
            "max_flagged_pixel_fraction": float(np.max(flagged)),
            "image_false_alarm_rate": float(np.mean(alarms)),
        }
    return out


def control_curve(
    good_probs: list[np.ndarray],
    grid: np.ndarray | None = None,
    min_area_frac: float = 1e-3,
) -> tuple[tuple[float, float, float], ...]:
    """(threshold, mean flagged fraction, image false-alarm rate) over a grid.

    Plotted against the risk curve this is the whole trade in one picture: risk
    falls as the threshold drops, false alarms rise.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    if not good_probs:
        return ()
    rows = []
    for t in grid:
        preds = [p >= float(t) for p in good_probs]
        rows.append((
            float(t),
            float(np.mean([flagged_fraction(q) for q in preds])),
            float(np.mean([image_flagged(q, min_area_frac) for q in preds])),
        ))
    return tuple(rows)
