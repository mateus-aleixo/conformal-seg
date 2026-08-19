"""Image preprocessing, shared by training and serving.

Deliberately **torch-free**: the serving container installs neither torch nor
torchvision (see `requirements/serve.txt`), so anything on the inference path has
to live here rather than in `data.py`.

There is a correctness reason as well as a size one. The conformal threshold is
fitted on probability maps produced from pixels that went through one specific
decode, resize and normalisation. Serve an image that took a different path, for
instance a PIL resize instead of openCV INTER_AREA, and the maps shift underneath
a threshold that was calibrated for the old ones. The guarantee is a statement
about a pipeline, not about a set of weights. One implementation, imported by both
sides, is how that stays true.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _bgr_to_rgb01(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def load_image(path: Path, size: int) -> np.ndarray:
    """Decode a file to RGB float32 [0,1], HWC, resized to `size`."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return _bgr_to_rgb01(bgr, size)


def decode_image(data: bytes, size: int) -> np.ndarray:
    """Decode raw bytes (an uploaded PNG/JPEG) the same way `load_image` does.

    Raises ValueError on anything openCV cannot decode, so the API can answer
    400 rather than 500 for a client that posts a text file.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image; expected PNG, JPEG or similar")
    return _bgr_to_rgb01(bgr, size)


def normalize_chw(rgb01: np.ndarray) -> np.ndarray:
    """HWC [0,1] -> CHW float32, ImageNet-normalised, ready for the model."""
    x = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).copy()
