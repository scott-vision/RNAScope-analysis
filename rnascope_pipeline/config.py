from __future__ import annotations

"""Configuration dataclass for the RNAscope pipeline.

This module defines :class:`Config`, which encapsulates all user adjustable
settings for running the pipeline.  The defaults correspond to the values used
in the original monolithic script.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration for the pipeline.

    Parameters mirror the table in the project README.  The configuration also
    exposes convenience properties for the various output sub‑directories.
    """

    # Paths
    root: Path = Path("../RNAScope/")
    """Project root containing ``Rat*`` folders and the ``maxima/`` directory."""

    out_dir: Path = Path("results_rnascope")
    """Destination directory for all pipeline outputs."""

    # Imaging parameters
    pixel_size_um: float = 0.1455
    """Microscopy pixel size in microns per pixel."""

    expansion_um: float = 2.5
    """Expansion distance for nuclei labels in microns."""

    dapi_index: int = 0
    """Channel index for the DAPI channel."""

    gob_index: int = 1
    """Channel index for the GoB channel."""

    goa_index: int = 2
    """Channel index for the GoA channel."""

    spot_marker_size: int = 1
    """Half size of the cross used to mark spots in overlay images."""

    load_saved_masks: bool = True
    """If ``True``, reuse previously computed Cellpose segmentation masks."""

    transpose_xy: bool = True
    """Swap ROI (``x, y``) coordinates to (``y, x``) if required."""

    skip_empty_roi: bool = True
    """Skip polygon ROIs that rasterise to an empty mask."""

    debug: bool = True
    """Enable verbose logging output."""

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------
    @property
    def cutouts_dir(self) -> Path:
        return self.out_dir / "cutouts"

    @property
    def qc_overlays_dir(self) -> Path:
        return self.out_dir / "qc_overlays"

    @property
    def masks_dir(self) -> Path:
        return self.out_dir / "masks"

    @property
    def csv_dir(self) -> Path:
        return self.out_dir / "csv"

    @property
    def roi_masks_dir(self) -> Path:
        return self.out_dir / "roi_masks_cropped"
