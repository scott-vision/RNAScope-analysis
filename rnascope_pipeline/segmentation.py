"""Cellpose based nuclei segmentation."""

from __future__ import annotations

import numpy as np
from cellpose import models


def create_model() -> models.CellposeModel:
    """Instantiate a default Cellpose model with GPU if available."""
    use_gpu = models.use_gpu()
    return models.CellposeModel(gpu=use_gpu)


def segment_nuclei(dapi_img: np.ndarray, model: models.CellposeModel) -> np.ndarray:
    """Segment nuclei from a 2‑D DAPI image using Cellpose."""
    masks, _flows, _styles = model.eval(dapi_img)
    return masks.astype(np.int32)
