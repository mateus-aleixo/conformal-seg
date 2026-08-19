"""Render the operating-point figure: what the guarantee costs on good parts.

Conformal risk control picks the threshold by reading ONE curve, the missed-defect
rate on defective parts, and walking down until the corrected risk meets alpha. It
never looks at defect-free parts, because they contain no defect pixels to miss.

This figure draws the curve it reads and the curve it ignores, on the same axis.
Both are rates in [0, 1], so they belong on one scale; the gap between them at the
chosen threshold is the whole engineering argument. The third panel is the same
category at 4x the pixel count, which separates "the model is bad" from "the target
is unreachable".

    python scripts/make_control_figure.py  ->  docs/figures/operating_point.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "figures" / "operating_point.png"

# Categorical slots 1 and 2 of the validated palette. Checked with the dataviz
# validator: all six checks pass on the light surface (worst adjacent CVD
# dE 24.7 protan, normal-vision dE 33.6).
RISK = "#2a78d6"
ALARM = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

# (run directory under runs/, panel title, verdict)
PANELS = [
    ("metal_nut", "metal_nut  ·  320 px",
     "one clean part in 22 escalated:\nthe guarantee is affordable"),
    ("grid", "grid  ·  320 px",
     "every clean part escalated:\nthe guarantee automates nothing"),
    ("grid_640", "grid  ·  640 px",
     "sharper model, same verdict:\nresolution was not the constraint"),
]


def panel(ax, run: str, title: str, verdict: str) -> None:
    calib = json.loads((ROOT / "runs" / run / "calibration.json").read_text())
    thr = calib["calibration"]["threshold"]
    alpha = calib["calibration"]["alpha"]
    risk_curve = calib["calibration"]["risk_curve"]
    control = calib["control"]
    ctrl_curve = calib["control_curve"]

    ax.plot([t for t, _ in risk_curve], [r for _, r in risk_curve],
            color=RISK, linewidth=2, solid_capstyle="round",
            label="missed-defect rate (what calibration reads)")
    ax.plot([r[0] for r in ctrl_curve], [r[2] for r in ctrl_curve],
            color=ALARM, linewidth=2, solid_capstyle="round",
            label="false-alarm rate on clean parts (what it ignores)")

    ax.axhline(alpha, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.995, alpha + 0.022, f"α = {alpha:.2f}", ha="right", va="bottom",
            fontsize=8, color=MUTED)

    far = control["at"]["conformal"]["image_false_alarm_rate"]
    area = control["at"]["conformal"]["mean_flagged_pixel_fraction"]

    ax.axvline(thr, color=MUTED, linewidth=1, zorder=1)
    # When the alarm curve pins to the top the annotation lives up there, so the
    # threshold label drops to the floor to keep out of its way.
    lab_y, lab_va = (0.02, "bottom") if far > 0.6 else (0.985, "top")
    ax.text(thr + 0.015, lab_y, f"λ̂ = {thr:.2f}", ha="left", va=lab_va,
            fontsize=8.5, color=INK)

    ax.plot([thr], [far], "o", markersize=8, color=ALARM,
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    # 0.016% of a frame is not "0.0%": rounding to zero would read as "costs nothing".
    area_txt = "<0.1%" if 0 < area < 0.001 else f"{area:.1%}"
    ax.annotate(f"{far:.0%} of clean parts flagged\n({area_txt} of the frame)",
                xy=(thr, far), xytext=(0.36, far + (0.15 if far < 0.6 else -0.28)),
                fontsize=8.2, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))

    ax.set_title(f"{title}\n{verdict}", fontsize=9.2, color=INK, pad=8,
                 linespacing=1.5)
    ax.set_xlabel("mask threshold λ", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4), sharey=True)
    for ax, (run, title, verdict) in zip(axes, PANELS, strict=True):
        panel(ax, run, title, verdict)
    axes[0].set_ylabel("rate", fontsize=9, color=MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.008))

    fig.suptitle("The curve conformal calibration reads, and the one it does not",
                 fontsize=12, y=0.98, color=INK)
    fig.text(0.5, 0.15,
             "Lowering λ to stop missing defects can only flag more pixels, so the two "
             "rates move in opposite directions, and calibration is blind to the orange "
             "one.\nQuadrupling grid's pixel count moved the naive operating point a long "
             "way, and the calibrated one not at all.",
             ha="center", va="top", fontsize=8.3, color=MUTED, linespacing=1.7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.19, 1, 0.93])
    fig.savefig(OUT, dpi=120, facecolor="white")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
