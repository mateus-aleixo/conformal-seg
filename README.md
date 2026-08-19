# conformal-seg

[![ci](https://github.com/mateus-aleixo/conformal-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/mateus-aleixo/conformal-seg/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Industrial defect segmentation whose masks carry a guarantee.** A torchvision
segmentation model is fine-tuned on MVTec AD defects and thresholded by **conformal
risk control**, so that the predicted defect region provably misses at most α of the
true defect pixels (pixel-level false-negative rate ≤ α, finite-sample and
distribution-free). The model is exported to ONNX with a parity check, and inference
runs torch-free.

Second of a series applying a single principle, *a prediction without a trustworthy
confidence statement is not a decision aid*, to three different kinds of data:

| repo | modality | the guarantee |
|---|---|---|
| [conformal-rul](https://github.com/mateus-aleixo/conformal-rul) | sensor sequences | RUL intervals with verified coverage, live on AWS Lambda |
| **conformal-seg** | vision | defect masks bounding the missed-defect rate |
| [conformal-rag](https://github.com/mateus-aleixo/conformal-rag) | language | selective QA that abstains at a calibrated error rate |

![Naive threshold vs conformal threshold on both categories](docs/figures/naive_vs_conformal.png)

*Top row: the method earning its keep. The calibrated mask is barely larger and
misses a third as much. Bottom row: the same method on a category the model never
learned, honouring its 10% promise by covering most of the frame. Both rows show the
test image closest to that category's held-out mean, not the best-looking one.*

## Results

Full numbers in [`docs/results.md`](docs/results.md). The short version:

| | naive 0.5 | conformal | verdict |
|---|---|---|---|
| `metal_nut` | IoU 0.691, **FNR 0.167 ✗** | IoU 0.644, **FNR 0.055 ✓** | the guarantee costs about 5 IoU points and fixes a 17% miss rate |
| `grid` | IoU 0.184, FNR 0.385 ✗ | IoU 0.015, **FNR 0.010 ✓**, **mask area 0.81** | the guarantee is honoured by flagging 81% of the image: valid, and useless |

`grid` is reported rather than buried. Conformal risk control guarantees *validity,
not utility*, and a category where the model fails is the clearest way to show what
the method does and does not buy you.

## Why put a guarantee on a mask

The industrial question is never "colour the defect pixels". It is **"can this part
ship without a human looking at it?"** A raw sigmoid threshold gives no answer,
because 0.5 means nothing off-distribution. Conformal risk control does. Pick the
mask threshold

λ̂ = largest λ such that (n/(n+1)) · R̂(λ) + 1/(n+1) ≤ α

where R̂(λ) is the mean fraction of true-defect pixels missed on a calibration split.
Under exchangeability the deployed masks then miss **at most α of defect pixels in
expectation**. This is the worked example of Angelopoulos et al. (2022), implemented
here end to end: automate confidently above the bar, route to a human below it. It is
the same decision shape as automated visual verification anywhere, namely act only
when the model is *provably* careful enough, and escalate the rest.

## Design

- **Data.** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  (Bergmann et al., CVPR 2019), categories `metal_nut` and `grid`. MVTec AD is an
  anomaly-detection benchmark, so defect masks exist only in its test split. This
  repo therefore follows the supervised protocol: the mask-annotated images are
  re-split 60/20/20 into train, calibration and test. That is stated plainly because
  the caveat is part of the method, namely that n is small, and the conformal
  guarantee is exactly the tool that stays valid at small n. The dataset is licensed
  **CC BY-NC-SA 4.0** (non-commercial research and portfolio use, attribution
  required, data never redistributed); `scripts/fetch_mvtec.py` downloads and
  extracts it.
- **Preprocessing.** openCV for decoding and resizing, pillow for mask handling and
  augmentation (flips, rotations), deterministic under a seed.
- **Model.** torchvision `deeplabv3_mobilenet_v3_large` with a pretrained backbone
  and a 1-channel defect head. The backbone is frozen by default so the fine-tune
  fits on a CPU overnight, and unfrozen with one flag on a GPU.
- **Calibration.** `conformal.py` implements conformal risk control over the
  threshold grid, the FNR-against-λ risk curve, and held-out verification. Same
  correction and same small-n honesty as the sibling repos.
- **Export.** ONNX (opset 18) plus a parity check with max |Δ| logged. `predict.py`
  runs on onnxruntime only, with no torch at inference, following the conformal-rul
  serving pattern.
- **Tests and CI.** The suite runs on synthetic fixtures (random ellipse "defects"),
  with no dataset, no pretrained weights and no network. Deterministic by
  construction.

## Quickstart

```bash
uv sync --all-extras                      # or: pip install -e ".[dev]"
uv run pytest                             # synthetic fixtures, CPU, no downloads
uv run python scripts/fetch_mvtec.py      # about 5.3 GB once; extracts 2 categories
uv run python -m conformal_seg.train --category metal_nut --unfreeze --lr 1e-4 --epochs 60
uv run python -m conformal_seg.calibrate --category metal_nut --alpha 0.1
uv run python -m conformal_seg.onnx_export --category metal_nut --check
uv run python -m conformal_seg.predict image.png --category metal_nut --mask out.png
```

**GPU note.** `pyproject.toml` pins CPU torch from PyPI, which is what CI wants. For
a CUDA build, install it into the venv and then invoke the venv's Python directly:
`uv run` re-syncs the environment from `pyproject.toml` on every call and will
silently put the CPU wheel back.

```bash
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.13.0+cu126" torchvision
./.venv/Scripts/python.exe -m conformal_seg.train --category metal_nut --device cuda
```

## Scope

Both categories are trained, calibrated, exported to ONNX and parity-checked, with
the evaluation tables in [`docs/results.md`](docs/results.md).

Known limits and the natural next steps: tiled or high-resolution inference, which is
what `grid` actually needs; a plotted risk curve; and a defect-free control split to
measure the false-alarm side of the trade.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, 2022.
- Bergmann, Fauser, Sattlegger, Steger, *MVTec AD: A Comprehensive Real-World
  Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

## License

MIT for the code. The dataset is governed by its own CC BY-NC-SA 4.0 terms.
Built by [Mateus Aleixo](https://github.com/mateus-aleixo).
