"""
Shared letterbox preprocessing for object and face detectors.

Both YOLOv8n (object) and YOLOv8n-Face (face) detectors resize input with
aspect-ratio preserved and pad to a square. This utility centralizes that
logic so it is implemented in exactly one place.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    img: np.ndarray, new_size: int
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize image with aspect-ratio preserved, pad to square.

    Args:
        img: Source BGR image (H, W, 3).
        new_size: Target square size (e.g. 640).

    Returns:
        (resized_and_padded_image, gain, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad_w, pad_h = (new_size - nw) / 2, (new_size - nh) / 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, r, (pad_w, pad_h)
