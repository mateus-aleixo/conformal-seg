"""MVTec AD supervised re-split + synthetic fixtures.

MVTec AD is an anomaly-detection benchmark: its train split is defect-free and
masks exist only under test/. The supervised protocol here takes every
mask-annotated defect image of a category and re-splits it 60/20/20 into
train / calibration / test by a seeded shuffle. Small n, stated openly — the
conformal calibration is exactly the tool that stays valid at small n.

openCV decodes and resizes images (BGR -> RGB); pillow handles masks. Both are
deliberate, visible choices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLITS = ("train", "cal", "test")


@dataclass(frozen=True)
class Item:
    image_path: Path
    mask_path: Path
    defect: str  # defect type folder name, e.g. "bent", "scratch"


def discover_items(root: Path, category: str) -> list[Item]:
    """All mask-annotated defect images of an MVTec category.

    Layout: <root>/<category>/test/<defect>/NNN.png with masks at
    <root>/<category>/ground_truth/<defect>/NNN_mask.png; 'good' has no masks.
    """
    cat = root / category
    items: list[Item] = []
    for defect_dir in sorted((cat / "test").iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        for img in sorted(defect_dir.glob("*.png")):
            mask = cat / "ground_truth" / defect_dir.name / f"{img.stem}_mask.png"
            if mask.exists():
                items.append(Item(img, mask, defect_dir.name))
    return items


def split_items(
    items: list[Item], seed: int = 17, fractions: tuple[float, float] = (0.6, 0.2)
) -> dict[str, list[Item]]:
    """Seeded 60/20/20 shuffle-split. Same seed -> same split, forever."""
    pool = list(items)
    random.Random(seed).shuffle(pool)
    n = len(pool)
    n_train = int(n * fractions[0])
    n_cal = int(n * fractions[1])
    return {
        "train": pool[:n_train],
        "cal": pool[n_train : n_train + n_cal],
        "test": pool[n_train + n_cal :],
    }


def load_pair(item: Item, size: int) -> tuple[np.ndarray, np.ndarray]:
    """image float32 [0,1] RGB HWC; mask uint8 {0,1} HW. openCV in, pillow for mask."""
    bgr = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(item.image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    mask_img = Image.open(item.mask_path).convert("L").resize((size, size), Image.NEAREST)
    mask = (np.asarray(mask_img) > 127).astype(np.uint8)
    return rgb.astype(np.float32) / 255.0, mask


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensors(rgb01: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = (rgb01 - _IMAGENET_MEAN) / _IMAGENET_STD
    return (
        torch.from_numpy(x.transpose(2, 0, 1).copy()),
        torch.from_numpy(mask.astype(np.float32))[None],
    )


def augment(rgb01: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Flips and quarter rotations — label-preserving for surface defects."""
    k = rng.randrange(4)
    if k:
        rgb01 = np.rot90(rgb01, k).copy()
        mask = np.rot90(mask, k).copy()
    if rng.random() < 0.5:
        rgb01 = np.fliplr(rgb01).copy()
        mask = np.fliplr(mask).copy()
    return rgb01, mask


class DefectSegDataset(Dataset):
    def __init__(self, items: list[Item], size: int = 320, train: bool = False, seed: int = 17):
        self.items = items
        self.size = size
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, mask = load_pair(self.items[i], self.size)
        if self.train:
            rgb, mask = augment(rgb, mask, self.rng)
        return to_tensors(rgb, mask)


# -- synthetic fixtures (tests / CI: no dataset, no network) -----------------

def make_synthetic_dir(root: Path, category: str = "synth", n: int = 12, size: int = 96,
                       seed: int = 5) -> Path:
    """Random ellipse 'defects' on noise, in the exact MVTec layout."""
    rng = np.random.default_rng(seed)
    img_dir = root / category / "test" / "blob"
    gt_dir = root / category / "ground_truth" / "blob"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = (rng.uniform(40, 90, (size, size, 3))).astype(np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (int(rng.uniform(20, size - 20)), int(rng.uniform(20, size - 20)))
        axes = (int(rng.uniform(6, 14)), int(rng.uniform(6, 14)))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        img[mask == 1] = (220, 60, 60)  # defect is visibly different: learnable
        cv2.imwrite(str(img_dir / f"{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        Image.fromarray(mask * 255).save(gt_dir / f"{i:03d}_mask.png")
    return root
