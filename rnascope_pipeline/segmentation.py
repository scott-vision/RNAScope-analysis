"""Cellpose based nuclei segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cellpose import models
from skimage.transform import resize


def create_model(model_path: Path | str | None = None) -> models.CellposeModel:
    """Instantiate a default Cellpose model with GPU if available."""
    use_gpu = True
    if model_path is not None:
        return models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    return models.CellposeModel(gpu=use_gpu)


def _resize_image(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if img.shape == shape:
        return np.asarray(img)
    out = resize(img, shape, order=1, preserve_range=True, anti_aliasing=True)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
    else:
        out = out.astype(img.dtype, copy=False)
    return np.asarray(out)


def _resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == shape:
        return labels.astype(np.int32)
    out = resize(labels, shape, order=0, preserve_range=True, anti_aliasing=False)
    return np.asarray(np.rint(out), dtype=np.int32)


def _segment_tiled_downsampled(
    dapi_img: np.ndarray,
    model: models.CellposeModel,
    *,
    patch_size: int,
    downsample: int,
) -> np.ndarray:
    source_tile = patch_size * downsample
    H, W = dapi_img.shape
    stitched = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    for y0 in range(0, H, source_tile):
        for x0 in range(0, W, source_tile):
            y1 = min(y0 + source_tile, H)
            x1 = min(x0 + source_tile, W)
            tile = np.asarray(dapi_img[y0:y1, x0:x1])
            model_tile = _resize_image(tile, (patch_size, patch_size))
            masks, _flows, _styles = model.eval(model_tile)
            labels = _resize_labels(np.asarray(masks), tile.shape)
            if labels.max() <= 0:
                continue

            relabeled = np.zeros(labels.shape, dtype=np.int32)
            for lab in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
                relabeled[labels == lab] = next_label
                next_label += 1
            stitched[y0:y1, x0:x1] = relabeled

    return stitched


def segment_nuclei(
    dapi_img: np.ndarray,
    model: models.CellposeModel,
    *,
    patch_size: int | None = None,
    downsample: int = 1,
) -> np.ndarray:
    """Segment nuclei from a 2-D DAPI image using Cellpose."""
    if patch_size is not None and downsample > 1:
        return _segment_tiled_downsampled(
            dapi_img,
            model,
            patch_size=patch_size,
            downsample=downsample,
        )
    masks, _flows, _styles = model.eval(dapi_img)
    return masks.astype(np.int32)
