"""torchvision deeplabv3_mobilenet_v3_large with a 1-channel defect head.

Pretrained backbone for real training; weights=None in tests so CI never touches
the network. Backbone frozen by default — head-only fine-tune fits a CPU
overnight; `--unfreeze` on a GPU trains everything.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


def build_kwargs(pretrained: bool) -> dict:
    """Constructor kwargs for the torchvision model.

    The pretrained VOC checkpoint was trained WITH an auxiliary head, and
    torchvision refuses to load it under `aux_loss=False`
    ("expected value True but got False"). So we must request aux_loss=True
    whenever we load those weights — and then discard the aux head, which we
    never use. Building from scratch has no such constraint.
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    return {"weights": weights, "aux_loss": True if pretrained else False}


class DefectSeg(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(**build_kwargs(pretrained))
        # swap the 21-class VOC head for a single defect logit
        self.net.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        # drop the auxiliary head: forward() skips it when it is None, so it
        # costs no compute and contributes no gradients
        self.net.aux_classifier = None
        if freeze_backbone:
            for p in self.net.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]  # (B, 1, H, W) logits


class ExportWrapper(nn.Module):
    """ONNX-friendly: tensor in, sigmoid probabilities out (no dict output)."""

    def __init__(self, model: DefectSeg):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice — BCE for pixel calibration, Dice against class imbalance."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)
