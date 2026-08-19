from pathlib import Path

import torch

from conformal_seg.model import (
    DefectSeg,
    ExportWrapper,
    build_kwargs,
    dice_loss,
    seg_loss,
)
from conformal_seg.train import train


def test_pretrained_requests_aux_loss():
    """Regression: the pretrained VOC checkpoint was trained with an auxiliary
    head and torchvision refuses to load it under aux_loss=False. Every test
    here builds with pretrained=False, so this argument logic is asserted
    directly — otherwise the failure only appears on a real training run
    (which is exactly how it was found)."""
    assert build_kwargs(pretrained=True)["aux_loss"] is True
    assert build_kwargs(pretrained=True)["weights"] is not None
    assert build_kwargs(pretrained=False)["aux_loss"] is False
    assert build_kwargs(pretrained=False)["weights"] is None


def test_aux_head_dropped():
    """The aux head must be gone: unused params would still take gradients."""
    assert DefectSeg(pretrained=False).net.aux_classifier is None


def test_forward_shape_and_head_swap():
    model = DefectSeg(pretrained=False, freeze_backbone=True)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)
    frozen = all(not p.requires_grad for p in model.net.backbone.parameters())
    assert frozen
    assert any(p.requires_grad for p in model.net.classifier.parameters())


def test_losses_behave():
    logits = torch.full((1, 1, 8, 8), 8.0)  # confident defect everywhere
    target_all = torch.ones(1, 1, 8, 8)
    target_none = torch.zeros(1, 1, 8, 8)
    assert dice_loss(logits, target_all) < 0.05
    assert seg_loss(logits, target_all) < seg_loss(logits, target_none)


def test_export_wrapper_outputs_probabilities():
    wrapped = ExportWrapper(DefectSeg(pretrained=False)).eval()
    with torch.no_grad():
        out = wrapped(torch.randn(1, 3, 64, 64))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_train_smoke_runs_and_checkpoints(synth_root, tmp_path):
    """Two capped epochs on synthetic data, CPU, random weights: the loop must
    run end to end, log JSONL, and write both checkpoints."""
    out = tmp_path / "run"
    best = train(
        data_root=Path(synth_root), category="synth", out_dir=out,
        epochs=2, batch_size=2, size=64, device="cpu",
        pretrained=False, max_steps=6,
    )
    assert best.exists()
    assert (out / "last.pt").exists()
    log = (out / "train_log.jsonl").read_text().strip().splitlines()
    assert len(log) >= 1
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    assert ckpt["category"] == "synth"
