"""Training loop. Seeded, checkpointed, JSONL-logged. CLI:

    python -m conformal_seg.train --category metal_nut --epochs 30
    python -m conformal_seg.train --category grid --unfreeze --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DefectSegDataset, discover_items, split_items
from .metrics import iou
from .model import DefectSeg, seg_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: DefectSeg, loader: DataLoader, device: str) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
        for p, t in zip(prob[:, 0], y.numpy()[:, 0], strict=True):
            scores.append(iou(p >= 0.5, t >= 0.5))
    return float(np.mean(scores)) if scores else 0.0


def train(
    data_root: Path,
    category: str,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    size: int = 320,
    seed: int = 17,
    device: str = "cpu",
    unfreeze: bool = False,
    pretrained: bool = True,
    max_steps: int | None = None,  # tests: cap work
) -> Path:
    seed_everything(seed)
    splits = split_items(discover_items(data_root, category), seed=seed)
    if not splits["train"]:
        raise SystemExit(f"no annotated defect images found under {data_root / category}")

    if len(splits["train"]) < batch_size:
        batch_size = max(2, len(splits["train"]))  # drop_last must leave >= 1 batch
    train_ds = DefectSegDataset(splits["train"], size=size, train=True, seed=seed)
    val_ds = DefectSegDataset(splits["cal"], size=size)  # cal doubles as val: never test
    # drop_last: a stray batch of 1 kills BatchNorm in DeepLab's ASPP pooling
    # branch ("Expected more than 1 value per channel"); with small defect sets
    # a size-1 remainder is common, so dropping it is the correct default.
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = DefectSeg(pretrained=pretrained, freeze_backbone=not unfreeze).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_iou, best_path = -1.0, out_dir / "best.pt"
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.perf_counter()
        for x, y in train_dl:
            opt.zero_grad()
            loss = seg_loss(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
            n_batches += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        sched.step()

        val_iou = evaluate(model, val_dl, device)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "loss": epoch_loss / max(n_batches, 1),
                "val_iou": val_iou, "lr": sched.get_last_lr()[0],
                "seconds": round(time.perf_counter() - t0, 1),
            }) + "\n")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"state_dict": model.state_dict(), "category": category,
                 "size": size, "seed": seed, "val_iou": val_iou}, best_path,
            )
        if max_steps is not None and step >= max_steps:
            break

    torch.save({"state_dict": model.state_dict(), "category": category,
                "size": size, "seed": seed, "val_iou": best_iou}, out_dir / "last.pt")
    return best_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    p.add_argument("--category", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--unfreeze", action="store_true")
    a = p.parse_args(argv)
    out = a.out or Path("runs") / a.category
    best = train(a.data_root, a.category, out, a.epochs, a.batch_size, a.lr,
                 a.size, a.seed, a.device, a.unfreeze)
    print(f"best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
