"""Image manipulation helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.draw import polygon
from skimage.segmentation import expand_labels
import tifffile as tiff

from .config import Config
from .utils import log


# ---------------------------------------------------------------------------
# ROI and image helpers
# ---------------------------------------------------------------------------

def roi_polygon_to_mask(roi, shape: Tuple[int, int], *, transpose_xy: bool = False) -> np.ndarray:
    """Return a boolean mask for an ImageJ polygon ROI."""
    H, W = shape
    coords_fn = getattr(roi, "coordinates", None)
    if not callable(coords_fn):
        raise ValueError("ROI has no polygon coordinates; convert ROI to polygon in ImageJ.")

    pts = coords_fn()
    if pts is None or len(pts) < 3:
        raise ValueError("ROI has insufficient coordinates (need >=3 points).")

    pts = np.asarray(pts, dtype=float)
    if transpose_xy:
        ys, xs = pts[:, 1], pts[:, 0]
    else:
        ys, xs = pts[:, 0], pts[:, 1]

    rr, cc = polygon(ys, xs, (H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True

    if mask.sum() == 0:
        raise ValueError("ROI rasterized to an empty mask. Check coordinates or transpose_xy setting.")
    return mask


def tight_crop_nd(arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop ``arr`` to the bounding box of ``mask`` (supports N‑D arrays)."""
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    sl = [slice(None)] * (arr.ndim - 2) + [slice(y0, y1), slice(x0, x1)]
    return arr[tuple(sl)], (y0, y1, x0, x1)


def to_uint8_vis(img2d: np.ndarray) -> np.ndarray:
    """Contrast stretch ``img2d`` to 8‑bit for visualisation."""
    lo, hi = np.percentile(img2d, (1, 99))
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.uint8)
    return np.clip((img2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def make_rgb_background(gray: np.ndarray) -> np.ndarray:
    """Stack ``gray`` into a 3‑channel RGB image."""
    return np.stack([gray, gray, gray], axis=-1)


def draw_crosses_inplace(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, color: tuple, size: int) -> None:
    """Draw coloured crosses onto ``rgb`` in‑place."""
    h, w, _ = rgb.shape
    r, g, b = color
    for x, y in zip(xs, ys):
        xi = int(x)
        yi = int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        x0, x1 = max(0, xi - size), min(w, xi + size + 1)
        rgb[yi, x0:x1, 0] = r
        rgb[yi, x0:x1, 1] = g
        rgb[yi, x0:x1, 2] = b
        y0, y1 = max(0, yi - size), min(h, yi + size + 1)
        rgb[y0:y1, xi, 0] = r
        rgb[y0:y1, xi, 1] = g
        rgb[y0:y1, xi, 2] = b


def save_tiff(path, arr) -> None:
    """Small wrapper around :func:`tifffile.imwrite` with uint16 cast."""
    tiff.imwrite(str(path), arr)


def expand_label_mask(labels: np.ndarray, distance_px: int, roi_mask: np.ndarray) -> np.ndarray:
    """Expand nucleus labels and clip to ``roi_mask``."""
    labels_exp = expand_labels(labels, distance=distance_px)
    labels_exp[~roi_mask] = 0
    return labels_exp
