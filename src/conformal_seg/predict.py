"""Torch-free inference: onnxruntime + the calibrated threshold.

    python -m conformal_seg.predict image.png --category metal_nut --mask out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict_mask(image_path: Path, onnx_path: Path, threshold: float, size: int = 320
                 ) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    prob = sess.run(None, {"image": x})[0][0, 0]
    return prob, (prob >= threshold).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--category", default="metal_nut")
    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--size", type=int, default=320)
    a = p.parse_args(argv)

    run_dir = Path("runs") / a.category
    calib = json.loads((run_dir / "calibration.json").read_text())
    thr = calib["calibration"]["threshold"]
    alpha = calib["calibration"]["alpha"]

    prob, mask = predict_mask(a.image, run_dir / "model.onnx", thr, a.size)
    frac = float(mask.mean())
    print(f"threshold {thr:.3f} (guarantees FNR <= {alpha} in expectation)")
    print(f"defect fraction: {frac:.4f} ({'defect region found' if frac > 0 else 'no defect at this confidence'})")
    if a.mask:
        Image.fromarray(mask * 255).save(a.mask)
        print(f"mask written: {a.mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
