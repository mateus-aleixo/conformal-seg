# Model card: conformal-seg

## What it is

Binary defect-segmentation networks for two MVTec AD categories (`metal_nut`
and `grid`), thresholded by conformal risk control so that the mask a caller
receives provably misses at most α of the true defect pixels, in expectation,
finite-sample and distribution-free. The served artifact is a pass/escalate
decision rather than a mask.

## Intended use

Benchmarking, teaching and demonstration of risk-controlled segmentation, and
as the reference implementation behind the public API in this repository. It is
**not** a certified inspection system. MVTec AD is a research benchmark of a few
hundred photographs per category under fixed lighting; nothing here has been
validated on a production line.

The `grid` model in particular is published as a **negative result** and must not
be read as fit for anything. See Limitations.

## Training data

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann,
Fauser, Sattlegger & Steger, CVPR 2019), categories `metal_nut` and `grid`,
downloaded by `scripts/fetch_mvtec.py`. Licensed **CC BY-NC-SA 4.0**: research
and portfolio use, attribution required, redistribution not permitted, so no
image is committed to this repository.

MVTec AD is an *anomaly-detection* benchmark, so pixel masks exist only in its
test split. This repo therefore follows a supervised protocol: every
mask-annotated defect image is re-split 60/20/20 into train / calibration / test
by a seeded shuffle (seed 17).

| category | train | calibration | test | defect-free control | defect types |
|---|---|---|---|---|---|
| `metal_nut` | 55 | 18 | 20 | 22 | bent, color, flip, scratch |
| `grid` | 34 | 11 | 12 | 21 | bent, broken, glue, metal contamination, thread |

n is small. That is the honest condition of the dataset under this protocol, and
it is the regime conformal calibration is built for: the guarantee is
finite-sample, not asymptotic.

The defect-free control images come from MVTec's untouched `test/good/` folder.
They are never trained or calibrated on. They exist to measure the side of the
trade the guarantee is blind to.

## Architecture and training

- torchvision `deeplabv3_mobilenet_v3_large`, pretrained backbone, 1-channel
  defect head, 320 px input (a 640 px `grid` variant is also reported).
- BCE + Dice loss, AdamW, cosine schedule, 60 epochs at lr 1e-4 with the
  backbone unfrozen. Frozen-backbone runs are reported for comparison.
- openCV decodes and resizes (INTER_AREA), pillow handles masks; flips and
  quarter rotations for augmentation, deterministic under the seed.
- Conformal layer: risk control over a threshold grid, taking the **largest**
  threshold whose finite-sample-corrected risk meets α. Two losses are
  selectable, both per-image, in [0, 1] and nondecreasing in the threshold, which
  is what the monotone construction of Angelopoulos et al. (2022) requires:
  `pixel` (fraction of true defect pixels missed) and `instance` (fraction of
  ground-truth connected components missed, counted found at 10% overlap).
- Export: ONNX opset 18 with a parity check against torch (max |Δ| 2.3e-06 and
  3.7e-06, gate at 1e-4). Inference is onnxruntime only, no torch.

## Metrics

Full tables in [results.md](results.md), at α = 0.10 on the held-out split.

| category | threshold | IoU | FNR | mask area | false alarms on clean parts |
|---|---|---|---|---|---|
| `metal_nut` | 0.500 naive | 0.691 | 0.167 ✗ | 0.173 | 0.000 |
| `metal_nut` | 0.130 conformal, pixel loss | 0.644 | **0.055 ✓** | 0.207 | **0.045** (1 of 22) |
| `metal_nut` | 0.290 conformal, instance loss | | **0.025 ✓** | 0.186 | **0.000** (0 of 22) |
| `grid` | 0.500 naive | 0.184 | 0.385 ✗ | | 0.429 |
| `grid` | 0.090 conformal | 0.015 | **0.010 ✓** | 0.814 | **1.000** (21 of 21) |

Both conformal rows honour α. Only one of them is deployable, which is the point
of reporting the last column.

## Limitations

- **The guarantee is one-sided by construction.** The loss counts what the mask
  *misses*, so a defect-free part contributes zero whatever the mask does.
  Nothing in the conformal machinery constrains false alarms, and the threshold
  it picks is lower than 0.5, which can only flag more. The defect-free control
  split exists because the theory offers no warning here.
- **Two losses are available and they bound different things.** `pixel` bounds
  the fraction of defect pixels missed; `instance` bounds the fraction of defect
  *instances* missed, counting an instance found when the mask covers at least
  10% of it. A threshold is meaningless without knowing which, so the
  calibration artifact records it. The 10% overlap is a product decision, like
  the escalation trigger, not a calibrated quantity.
- **`grid` meets its guarantee and is useless, under every variation tried.** It
  escalates every defect-free part. Retraining at 640 px raised validation IoU
  from 0.068 to 0.359 and cut naive false alarms from 0.429 to 0.095 without
  moving the calibrated rate at all. Switching to the instance loss shrank the
  mask sevenfold, 0.734 to 0.105 of the frame, and still did not move it. At λ̂
  the flagged-area distributions overlap (median 0.038 on defect images against
  0.020 on clean ones) and no escalation trigger separates them. The model has no
  discriminative signal on this category; 34 training images of a low-contrast
  texture is not enough. **Do not deploy the `grid` model for anything.**
- **Small n.** 11 to 18 calibration images per category. The guarantee remains
  valid, but the finite-sample correction is correspondingly loose and the
  measured rates carry wide binomial intervals.
- **Exchangeability is assumed within a category.** The re-split is random over
  one category's defect images under fixed capture conditions. Nothing supports
  transfer across categories, lighting, or cameras.
- **Per category, per resolution.** A threshold is valid only for the model,
  category and input size it was fitted on. The serving registry reads the input
  resolution out of the ONNX graph rather than trusting a flag, precisely because
  a mismatch here is silent.
- The escalation trigger (`min_area_frac`, default 0.1% of the frame) is a
  product decision, not a calibrated quantity. It is exposed as a request
  parameter and reported on every response.

## Safety framing

The decision this model supports is "can this part ship without a human looking
at it?" Getting that wrong in one direction ships a defect; in the other it
stops a line. Conformal risk control bounds only the first, so the API returns
α **and** the measured false-alarm rate on every response, and `GET /models`
reports both before any image is sent.

Any real deployment would additionally need: a defect-free control set drawn
from the actual line rather than a benchmark, periodic recalibration as the
process drifts, and a monitored escalation queue, since a system that escalates
more than a human reviewer can absorb has failed regardless of what it proves.
