"""Fit the conformal threshold on the calibration split; report held-out risk.

    python -m conformal_seg.calibrate --category metal_nut --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .conformal import calibrate, control_curve, control_report, held_out_report
from .data import (
    DefectSegDataset,
    GoodImageDataset,
    discover_good_items,
    discover_items,
    split_items,
)
from .model import DefectSeg


@torch.no_grad()
def predict_probs(model: DefectSeg, ds: DefectSegDataset, device: str) -> tuple[list, list]:
    model.eval()
    probs, masks = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        p = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        probs.append(p)
        masks.append(y[0].numpy() >= 0.5)
    return probs, masks


@torch.no_grad()
def predict_probs_unlabelled(model: DefectSeg, ds: GoodImageDataset, device: str) -> list:
    """Sigmoid maps for images that have no mask (the defect-free control)."""
    model.eval()
    return [
        torch.sigmoid(model(ds[i][None].to(device)))[0, 0].cpu().numpy()
        for i in range(len(ds))
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    p.add_argument("--category", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-area-frac", type=float, default=1e-3,
                   help="flagged area above which a part is escalated to a human")
    a = p.parse_args(argv)

    ckpt_path = a.checkpoint or Path("runs") / a.category / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=a.device, weights_only=True)
    model = DefectSeg(pretrained=False).to(a.device)
    model.load_state_dict(ckpt["state_dict"])

    splits = split_items(discover_items(a.data_root, a.category), seed=a.seed)
    cal_probs, cal_masks = predict_probs(model, DefectSegDataset(splits["cal"], a.size), a.device)
    cal = calibrate(cal_probs, cal_masks, alpha=a.alpha)

    test_probs, test_masks = predict_probs(model, DefectSegDataset(splits["test"], a.size), a.device)
    report = held_out_report(test_probs, test_masks, cal)

    # Defect-free control: what the guarantee costs on parts that are fine.
    good_paths = discover_good_items(a.data_root, a.category)
    good_probs = predict_probs_unlabelled(
        model, GoodImageDataset(good_paths, a.size), a.device
    )
    control = control_report(
        good_probs,
        {"naive": 0.5, "conformal": cal.threshold},
        min_area_frac=a.min_area_frac,
    )
    curve = control_curve(good_probs, min_area_frac=a.min_area_frac)

    out = a.out or Path("runs") / a.category / "calibration.json"
    out.write_text(json.dumps({
        "calibration": cal.to_dict(),
        "held_out": report,
        "control": control,
        "control_curve": [list(r) for r in curve],
    }, indent=2))
    print(f"threshold {cal.threshold:.3f} (alpha={a.alpha}, n_cal={cal.n})")
    print(f"held-out FNR mean {report['held_out_fnr_mean']:.4f} "
          f"(target <= {a.alpha}), mask fraction {report['mean_predicted_mask_fraction']:.3f}")
    if control["n_control"]:
        naive, conf = control["at"]["naive"], control["at"]["conformal"]
        print(f"control ({control['n_control']} defect-free parts): "
              f"false-alarm rate {naive['image_false_alarm_rate']:.3f} at 0.500 "
              f"-> {conf['image_false_alarm_rate']:.3f} at {cal.threshold:.3f}; "
              f"flagged area {naive['mean_flagged_pixel_fraction']:.4f} "
              f"-> {conf['mean_flagged_pixel_fraction']:.4f}")
    else:
        print("control: no defect-free images found, skipped")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
