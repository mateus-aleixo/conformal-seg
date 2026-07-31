"""ONNX export with a parity check — the conformal-rul serving pattern.

    python -m conformal_seg.onnx_export --category metal_nut --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import DefectSeg, ExportWrapper


def export(checkpoint: Path, out: Path, size: int = 320, check: bool = True) -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DefectSeg(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    wrapped = ExportWrapper(model).eval()

    dummy = torch.randn(1, 3, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, str(out),
        input_names=["image"], output_names=["prob"],
        dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=18,  # torch's dynamo exporter implements >= 18
    )

    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            ref = wrapped(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        max_diff = float(np.abs(ref - got).max())
        print(f"parity max |diff| = {max_diff:.2e}")
        if max_diff > 1e-4:
            raise SystemExit(f"parity check FAILED: {max_diff:.2e} > 1e-4")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    ckpt = a.checkpoint or Path("runs") / a.category / "best.pt"
    out = a.out or Path("runs") / a.category / "model.onnx"
    print(f"exported: {export(ckpt, out, a.size, a.check)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
