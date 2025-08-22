"""Utility helpers for the RNAscope pipeline."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import roifile

from .config import Config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(cfg: Config, msg: str) -> None:
    """Print ``msg`` if debugging is enabled."""
    if cfg.debug:
        print(msg)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: Config) -> None:
    """Create the standard output directories."""
    cfg.cutouts_dir.mkdir(parents=True, exist_ok=True)
    cfg.qc_overlays_dir.mkdir(parents=True, exist_ok=True)
    cfg.masks_dir.mkdir(parents=True, exist_ok=True)
    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    cfg.roi_masks_dir.mkdir(parents=True, exist_ok=True)


def empty_dir(d: Path) -> None:
    """Remove all contents of ``d`` while preserving the directory."""
    if d.exists():
        for p in d.iterdir():
            if p.is_file() or p.is_symlink():
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    else:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image / ROI discovery
# ---------------------------------------------------------------------------

def find_images(exp_dir: Path, keyword: str) -> List[Path]:
    """Return list of top‑level images under ``exp_dir`` containing ``keyword``."""
    k = keyword.lower()
    return [
        p
        for p in exp_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and k in p.name.lower()
    ]


def get_single_image(exp_dir: Path, keyword: str) -> Path:
    """Return exactly one matching top‑level image path; raise if 0 or >1."""
    images = find_images(exp_dir, keyword)
    if not images:
        raise RuntimeError(f"No '{keyword}' image found directly under {exp_dir}")
    if len(images) > 1:
        msg = "".join(f"  - {p}" for p in images)
        raise RuntimeError(
            f"Expected exactly one top-level '{keyword}' image in {exp_dir}, found {len(images)}:{msg}"
        )
    return images[0]


def read_points_roi(roi_path: Path | None, transpose_xy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return point coordinates from an ImageJ point ROI file."""
    if roi_path is None:
        return np.array([], dtype=int), np.array([], dtype=int)

    roi = roifile.roiread(str(roi_path))
    pts = roi.coordinates()
    if pts is None or len(pts) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    pts = np.asarray(pts)
    if transpose_xy:
        ys, xs = pts[:, 1], pts[:, 0]
    else:
        ys, xs = pts[:, 0], pts[:, 1]
    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)
    return xs, ys


def find_maxima_files(maxima_dir: Path, animal: str, region: str) -> Dict[str, Path | None]:
    """Locate GOA and GOB maxima ROI files for ``animal`` and ``region``."""
    patt = re.compile(rf"^{re.escape(animal)}_{re.escape(region)}_([a-zA-Z]+)_maxima\.roi$")
    out: Dict[str, Path | None] = {"GOA": None, "GOB": None}
    for p in maxima_dir.glob(f"{animal}_{region}_*_maxima.roi"):
        m = patt.match(p.name)
        if not m:
            continue
        tag = m.group(1).lower()
        if "goa" in tag:
            out["GOA"] = p
        elif "gob" in tag:
            out["GOB"] = p
    return out
