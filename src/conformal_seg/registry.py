"""Assemble a serving registry from training runs.

`runs/<name>/` is a training directory: checkpoints, logs, an ONNX export and a
calibration report, most of it useless to a server. `models/<category>/` is the
serving contract: the network, and the numbers a caller needs to interpret its
output.

    python -m conformal_seg.registry --run metal_nut
    python -m conformal_seg.registry --run grid_640 --category grid

Unlike conformal-rul, this registry is **not committed**. rul's exported networks
are a few hundred KB; this one carries a MobileNetV3 backbone at 42 MB per
category, which does not belong in git. Build it locally, and the Dockerfile bakes
whatever it finds into the image.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SERVING_MANIFEST = "serving.json"


def input_size(onnx_path: Path) -> int:
    """Read the spatial size the network was exported at, from the graph itself.

    Taking it from a CLI flag invites the one bug that matters here: serving at a
    different resolution than the threshold was calibrated for, silently.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape  # [batch, 3, H, W]
    h, w = shape[2], shape[3]
    if not isinstance(h, int) or not isinstance(w, int) or h != w:
        raise ValueError(f"expected a square static input, got {shape}")
    return h


def build(run_dir: Path, out_dir: Path, category: str | None = None) -> Path:
    """Copy the network and distil the calibration report into a manifest."""
    onnx = run_dir / "model.onnx"
    calib_path = run_dir / "calibration.json"
    for required in (onnx, calib_path):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} missing; run onnx_export and calibrate for {run_dir.name} first"
            )

    calib = json.loads(calib_path.read_text())
    cal, held = calib["calibration"], calib["held_out"]
    control = calib.get("control", {})
    conformal_control = control.get("at", {}).get("conformal", {})

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, out_dir / "model.onnx")
    # torch writes initializers beside the graph when they exceed the protobuf
    # limit; the graph references the file by relative name, so it travels with it.
    external = onnx.with_suffix(".onnx.data")
    if external.exists():
        shutil.copy2(external, out_dir / external.name)

    manifest = {
        "category": category or run_dir.name,
        "source_run": run_dir.name,
        "input_size": input_size(onnx),
        "threshold": cal["threshold"],
        "alpha": cal["alpha"],
        "n_calibration": cal["n"],
        "held_out": {
            "fnr_mean": held["held_out_fnr_mean"],
            "mask_fraction": held["mean_predicted_mask_fraction"],
            "n_test": held["n_test"],
        },
        "control": {
            "n": control.get("n_control", 0),
            "min_area_frac": control.get("min_area_frac", 1e-3),
            "false_alarm_rate": conformal_control.get("image_false_alarm_rate"),
            "flagged_fraction": conformal_control.get("mean_flagged_pixel_fraction"),
        },
    }
    (out_dir / SERVING_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="directory name under runs/")
    p.add_argument("--category", default=None,
                   help="name to serve it as; defaults to the run name")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--models-root", type=Path, default=Path("models"))
    a = p.parse_args(argv)

    category = a.category or a.run
    out = build(a.runs_root / a.run, a.models_root / category, category)
    manifest = json.loads((out / SERVING_MANIFEST).read_text())
    print(f"registry: {out}")
    print(f"  {manifest['category']} at {manifest['input_size']} px, "
          f"threshold {manifest['threshold']:.3f} (alpha {manifest['alpha']})")
    far = manifest["control"]["false_alarm_rate"]
    if far is not None:
        print(f"  escalates {far:.1%} of defect-free parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
