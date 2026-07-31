# conformal-seg

**Industrial defect segmentation whose masks carry a guarantee.** A torchvision
segmentation model fine-tuned on MVTec AD defects, thresholded by **conformal risk
control** so that the predicted defect region provably misses at most α of the true
defect pixels (pixel-level false-negative rate ≤ α, finite-sample,
distribution-free). Exported to ONNX with a parity check; inference runs torch-free.

Third of a series — one thesis, three modalities:

| repo | modality | the guarantee |
|---|---|---|
| [conformal-rul](https://github.com/mateus-aleixo/conformal-rul) | sensor sequences | RUL intervals with verified coverage, live on AWS Lambda |
| [conformal-rag](https://github.com/mateus-aleixo/conformal-rag) | language | selective QA that abstains at a calibrated error rate |
| **conformal-seg** | vision | defect masks bounding the missed-defect rate |

*A prediction without a trustworthy confidence statement is not a decision aid.*

> **Status: day one.** README pushed before the code, on purpose; the git log is the
> honest record. Trained results land in `docs/results.md` when the training runs
> have actually run — not before.

## Why a guarantee on a mask

The industrial question is never "colour the defect pixels", it is **"can this part
ship without a human looking at it?"** A raw sigmoid threshold gives no answer: 0.5
means nothing off-distribution. Conformal risk control does. Pick the mask threshold

λ̂ = largest λ such that (n/(n+1)) · R̂(λ) + 1/(n+1) ≤ α

where R̂(λ) is the mean fraction of true-defect pixels missed on a calibration split.
Under exchangeability, the deployed masks then miss **at most α of defect pixels in
expectation** — the exact worked example of Angelopoulos et al. (2022), implemented
here end to end: automate confidently above the bar, route to a human below it. The
same decision shape as automated visual verification anywhere: act only when the
model is *provably* careful enough, escalate the rest.

## Design

- **Data:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  (Bergmann et al., CVPR 2019), categories `metal_nut` and `grid`. MVTec AD is an
  anomaly-detection benchmark — defect masks exist only in its test split — so this
  repo follows the supervised protocol: the mask-annotated images are re-split
  60/20/20 into train/calibration/test, stated plainly here because the honest
  caveat is part of the method: n is small, and the conformal guarantee is exactly
  the tool that stays valid at small n.
  Licence **CC BY-NC-SA 4.0** — non-commercial research/portfolio use, attribution,
  data never redistributed; `scripts/fetch_mvtec.py` downloads and extracts.
- **Preprocessing:** openCV for decoding/resizing, pillow for mask handling and
  augmentation (flips, rotations) — deterministic under a seed.
- **Model:** torchvision `deeplabv3_mobilenet_v3_large`, pretrained backbone,
  1-channel defect head. Backbone frozen by default so the fine-tune fits a CPU
  overnight; unfrozen with one flag on a GPU.
- **Calibration:** `conformal.py` — conformal risk control over the threshold grid,
  FNR-vs-λ risk curve, held-out verification. Same correction, same small-n
  honesty as the sibling repos.
- **Export:** ONNX (opset 17) + parity check (max |Δ| logged); `predict.py` runs on
  onnxruntime only, no torch at inference — the conformal-rul serving pattern.
- **Tests/CI:** the suite runs on synthetic fixtures (random ellipse "defects"),
  no dataset, no pretrained weights, no network. Deterministic by construction.

## Quickstart

```bash
uv sync --all-extras                      # or: pip install -e ".[dev]"
uv run pytest                             # synthetic fixtures, CPU, no downloads
uv run python scripts/fetch_mvtec.py      # ~5.3 GB once; extracts 2 categories
uv run python -m conformal_seg.train --category metal_nut
uv run python -m conformal_seg.calibrate --alpha 0.1
uv run python -m conformal_seg.onnx_export --check
uv run python -m conformal_seg.predict image.png --mask out.png
```

## Roadmap

| Day | Deliverable |
|---|---|
| D1 | This README, scaffold, fetch script, synthetic-fixture test suite, CI |
| D2–3 | Fine-tune on `metal_nut` (overfit one batch first, then train); IoU/PR reported |
| D4 | Second category; eval tables in `docs/results.md` |
| D5 | Conformal masks: λ̂ per category, held-out FNR table, risk curve |
| D6 | ONNX export + parity; torch-free `predict.py` |
| D7+ | Polish; honest failure notes |

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control*, 2022.
- Bergmann, Fauser, Sattlegger, Steger — *MVTec AD: A Comprehensive Real-World
  Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

MIT licence (code). Dataset under its own CC BY-NC-SA 4.0 terms.
Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
