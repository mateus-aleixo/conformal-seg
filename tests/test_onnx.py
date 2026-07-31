from pathlib import Path

import numpy as np
import torch

from conformal_seg.model import DefectSeg
from conformal_seg.onnx_export import export


def test_export_parity_small(tmp_path):
    """Random-weight model, 64px: export must produce an ONNX file whose output
    matches torch to 1e-4 — the same parity bar conformal-rul ships with."""
    model = DefectSeg(pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": model.state_dict(), "category": "synth",
                "size": 64, "seed": 0, "val_iou": 0.0}, ckpt)
    out = export(ckpt, tmp_path / "model.onnx", size=64, check=True)  # raises on fail
    assert out.exists() and out.stat().st_size > 100_000


def test_predict_roundtrip(tmp_path, synth_root):
    """Torch-free path: onnx model + threshold produce a mask for a real file."""
    from conformal_seg.data import discover_items
    from conformal_seg.predict import predict_mask

    model = DefectSeg(pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": model.state_dict(), "category": "synth",
                "size": 64, "seed": 0, "val_iou": 0.0}, ckpt)
    onnx_path = export(ckpt, tmp_path / "model.onnx", size=64, check=False)

    image = discover_items(synth_root, "synth")[0].image_path
    prob, mask = predict_mask(image, onnx_path, threshold=0.5, size=64)
    assert prob.shape == (64, 64)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
