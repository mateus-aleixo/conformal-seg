# Results

Produced on code at `3ca4d33` (plus the `aux_loss` and ONNX fixes made
during this run). Hardware: RTX 3060 Laptop, 6 GB, CUDA 12.6, torch 2.13.
Reproduce: `scripts/fetch_mvtec.py` → `train` → `calibrate` → `onnx_export`.

## The headline

**Conformal risk control delivers exactly what it promises, and nothing more:
validity, not usefulness.** On a model that works, it costs ~5 IoU points and fixes
a miss rate that would otherwise have broken the 10% promise. On a model that
doesn't, it still honours the promise — by flagging 81% of every image.

Both cases are below. The second one is the reason this file exists.

## Setup

Supervised re-split of the mask-annotated defect images, seed 17, 60/20/20:

| category | train | calibration | test | defect types |
|---|---|---|---|---|
| `metal_nut` | 55 | 18 | 20 | bent, color, flip, scratch |
| `grid` | 34 | 11 | 12 | bent, broken, glue, metal_contamination, thread |

n is small. That is the honest condition of the dataset under a supervised
protocol, and it is precisely the regime conformal calibration is built for —
the guarantee is finite-sample, not asymptotic.

## Training

`deeplabv3_mobilenet_v3_large`, pretrained backbone, 1-channel head, 320 px,
BCE + Dice, AdamW + cosine. Best validation IoU (validation = the calibration
split; the test split is untouched until the table after this one):

| category | backbone | epochs | lr | best val IoU |
|---|---|---|---|---|
| `metal_nut` | frozen | 40 | 1e-3 | 0.442 |
| **`metal_nut`** | **unfrozen** | **60** | **1e-4** | **0.621** |
| `grid` | frozen | 40 | 1e-3 | 0.048 |
| **`grid`** | **unfrozen** | **60** | **1e-4** | **0.068** |

Unfreezing the backbone is worth ~0.18 IoU on `metal_nut` and nothing that
matters on `grid`. Roughly 1.6 s/epoch — the whole table is under 6 minutes of
compute, which is a fair description of how small this dataset is.

## Conformal calibration (α = 0.10), on the held-out test split

Threshold fitted on the calibration split only; every number below is from data
the threshold never saw.

### `metal_nut` — the guarantee is worth having

| threshold | IoU | precision | recall | **FNR** | mask area |
|---|---|---|---|---|---|
| 0.500 (naive) | **0.691** | 0.818 | 0.833 | **0.167 ✗** | 0.173 |
| 0.130 (conformal) | 0.644 | 0.672 | 0.945 | **0.055 ✓** | 0.207 |

The naive threshold looks better on IoU and **misses 16.7% of defect pixels** — it
would have violated a 10% promise nobody checked. The conformal threshold gives up
4.7 IoU points and 15 points of precision to buy a verified miss rate of 5.5%, with
masks 20% larger. For "can this part ship unseen?", that is the correct trade.

### `grid` — the guarantee holds and the output is useless

| threshold | IoU | precision | recall | **FNR** | mask area |
|---|---|---|---|---|---|
| 0.500 (naive) | 0.184 | 0.292 | 0.615 | 0.385 ✗ | 0.024 |
| 0.090 (conformal) | 0.015 | 0.015 | 0.990 | **0.010 ✓** | **0.814** |

The model never learned `grid` — thin, low-contrast texture defects, 34 training
images, downsampled to 320 px. Calibration did its job perfectly and the result is
absurd: to miss at most 10% of defect pixels with a model this weak, the only
admissible threshold flags **81% of the image**. FNR 0.0096, IoU 0.015.

This is the point worth taking away. Conformal risk control is a *validity*
guarantee, not a quality guarantee: it will always find a threshold that honours α,
and when the underlying model is bad that threshold is "flag almost everything".
The correct reading of a `grid` deployment is not "we have a guarantee" but
"mask area 0.81 means there is no usable signal here — fix the model, or don't
automate this category". Reporting mask area beside FNR is what makes that legible;
a paper reporting only coverage would have called this a success.

What would actually fix `grid`: higher resolution or tiled inference (its defects
are a few pixels wide), more annotated data, and a texture-anomaly approach rather
than supervised segmentation — which is what MVTec AD is designed for in the first
place.

## ONNX export

Parity against torch on random inputs, max |Δ|:

| category | max abs diff | file |
|---|---|---|
| `metal_nut` | 2.26e-06 | 493 KB |
| `grid` | 3.70e-06 | 493 KB |

Both under the 1e-4 gate. Inference path is onnxruntime only — no torch at serving
time, same pattern as [conformal-rul](https://github.com/mateus-aleixo/conformal-rul).

## Bugs this run found

Two, both invisible to the test suite until real data and real weights showed up:

1. **`aux_loss`** — the pretrained VOC checkpoint was trained with an auxiliary
   head and torchvision refuses to load it under `aux_loss=False`. Every test built
   with `pretrained=False`, so the failure only appeared on the first real training
   run. Fixed, and the argument logic is now asserted directly in a CI-safe test.
2. **ONNX export crashed on Windows** — torch's exporter prints a ✅ progress banner
   that a cp1252 console cannot encode. Fixed with `verbose=False` plus a stdout
   reconfigure.
