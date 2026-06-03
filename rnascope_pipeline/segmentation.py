"""Cellpose based nuclei segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cellpose import models


def create_model(model_path: Path | str | None = None) -> models.CellposeModel:
    """Instantiate a default Cellpose model with GPU if available."""
    use_gpu = True
    if model_path is not None:
        return models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    return models.CellposeModel(gpu=use_gpu)


def segment_nuclei(dapi_img: np.ndarray, model: models.CellposeModel) -> np.ndarray:
    """Segment nuclei from a 2-D DAPI image using Cellpose."""
    masks, _flows, _styles = model.eval(dapi_img)
    return masks.astype(np.int32)
