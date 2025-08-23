"""Per‑ROI analysis utilities."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import tifffile as tiff
from skimage.io import imsave

from .config import Config
from .image_utils import (
    draw_crosses_inplace,
    expand_label_mask,
    apply_orange_hot_lut,
    apply_red_lut,
    tight_crop_nd,
    to_uint8_vis,
)
from .segmentation import segment_nuclei
from .utils import log

# Grading bins (no bin 4, per request)
BIN_LABELS = [0, 1, 2, 3, 5]


def count_to_bin(c: int) -> int:
    if c <= 0:
        return 0
    if 1 <= c <= 3:
        return 1
    if 4 <= c <= 9:
        return 2
    if 10 <= c <= 15:
        return 3
    return 5


def summarize_bins(values) -> Dict[int, int]:
    out = {b: 0 for b in BIN_LABELS}
    for v in values:
        out[count_to_bin(int(v))] += 1
    return out


def analyze_roi(
    cfg: Config,
    animal: str,
    region: str,
    roi_name: str,
    full_img: np.ndarray,
    roi_mask: np.ndarray,
    gob_xy: Tuple[np.ndarray, np.ndarray],
    goa_xy: Tuple[np.ndarray, np.ndarray],
    model,
    expand_px: int,
) -> Tuple[List[Dict], Dict]:
    """Analyse a single polygon ROI and return per‑nucleus and per‑ROI metrics."""
    t0 = time.perf_counter()
    log(cfg, f"  > ROI '{roi_name}': starting analysis")

    # Prepare cut‑out and free‑form mask
    cutout, (y0, y1, x0, x1) = tight_crop_nd(full_img, roi_mask)
    mask_c = roi_mask[y0:y1, x0:x1]
    cutout_masked = cutout * mask_c[..., None]
    log(
        cfg,
        f"    cutout bbox=({y0}:{y1},{x0}:{x1}) size={cutout.shape}, "
        f"masked_size={cutout_masked.shape}",
    )

    # Save masked cut‑out
    cutout_name = f"{animal}_{region}_{roi_name}_cutout.tif"
    tiff.imwrite(str(cfg.cutouts_dir / cutout_name), cutout_masked)
    roi_mask_name = f"{animal}_{region}_{roi_name}_roi_mask_cropped.tif"
    tiff.imwrite(str(cfg.roi_masks_dir / roi_mask_name), (mask_c.astype(np.uint8) * 255))

    # DAPI channel for segmentation
    dapi = cutout_masked[..., cfg.dapi_index]

    # Select maxima inside ROI & convert to local coords
    def in_roi_and_local(x_all, y_all):
        if len(x_all) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        H, W = roi_mask.shape
        x_all = np.asarray(x_all)
        y_all = np.asarray(y_all)
        valid = (x_all >= 0) & (x_all < W) & (y_all >= 0) & (y_all < H)
        x_all = x_all[valid]
        y_all = y_all[valid]
        inside = roi_mask[y_all, x_all]
        x_roi = x_all[inside] - x0
        y_roi = y_all[inside] - y0
        return x_roi, y_roi

    gob_x_all, gob_y_all = gob_xy
    goa_x_all, goa_y_all = goa_xy
    gob_x, gob_y = in_roi_and_local(gob_x_all, gob_y_all)
    goa_x, goa_y = in_roi_and_local(goa_x_all, goa_y_all)
    log(cfg, f"    maxima counts in ROI: GOA={len(goa_x)}, GOB={len(gob_x)}")

    # Save separate GOA and GOB overlays
    goa_bg = to_uint8_vis(cutout_masked[..., cfg.goa_index])
    gob_bg = to_uint8_vis(cutout_masked[..., cfg.gob_index])
    goa_overlay = apply_red_lut(goa_bg)
    gob_overlay = apply_orange_hot_lut(gob_bg)
    draw_crosses_inplace(goa_overlay, goa_x, goa_y, color=(0, 255, 255), size=cfg.spot_marker_size)
    draw_crosses_inplace(gob_overlay, gob_x, gob_y, color=(0, 255, 255), size=cfg.spot_marker_size)
    overlay_name_goa = f"{animal}_{region}_{roi_name}_goa_maxima.png"
    overlay_name_gob = f"{animal}_{region}_{roi_name}_gob_maxima.png"
    imsave(str(cfg.qc_overlays_dir / overlay_name_goa), goa_overlay)
    imsave(str(cfg.qc_overlays_dir / overlay_name_gob), gob_overlay)
    log(cfg, f"    saved GOA overlay → {cfg.qc_overlays_dir / overlay_name_goa}")
    log(cfg, f"    saved GOB overlay → {cfg.qc_overlays_dir / overlay_name_gob}")

    # Segment nuclei
    mask_path = cfg.masks_dir / f"{animal}_{region}_{roi_name}_labels.tif"
    if cfg.load_saved_masks and mask_path.exists():
        labels = tiff.imread(str(mask_path)).astype(np.int32)
        log(cfg, f"    loaded labels from cache: {mask_path.name}")
    else:
        labels = segment_nuclei(dapi, model)
        labels[~mask_c] = 0
        tiff.imwrite(str(mask_path), labels.astype(np.uint16))
        log(cfg, f"    computed & saved labels: n_nuclei={int(labels.max())}")

    n_nuc = int(labels.max())
    labels_exp = expand_label_mask(labels, distance_px=expand_px, roi_mask=mask_c)
    log(cfg, f"    expanded labels by {expand_px}px (~{cfg.expansion_um} µm)")

    # Map spots to nuclei
    Hc, Wc = labels.shape

    def clamp_xy(x, y, W, H):
        x = np.clip(np.asarray(x), 0, W - 1)
        y = np.clip(np.asarray(y), 0, H - 1)
        return x, y

    gob_xc, gob_yc = clamp_xy(gob_x, gob_y, Wc, Hc)
    goa_xc, goa_yc = clamp_xy(goa_x, goa_y, Wc, Hc)

    gob_lab_nuc = labels[gob_yc, gob_xc]
    gob_lab_exp = labels_exp[gob_yc, gob_xc]
    goa_lab_nuc = labels[goa_yc, goa_xc]
    goa_lab_exp = labels_exp[goa_yc, goa_xc]

    # ROI area & densities
    roi_area_um2 = float(roi_mask.sum() * (cfg.pixel_size_um ** 2))
    gob_total = int(len(gob_x))
    goa_total = int(len(goa_x))
    gob_density = (gob_total / roi_area_um2) if roi_area_um2 > 0 else 0.0
    goa_density = (goa_total / roi_area_um2) if roi_area_um2 > 0 else 0.0
    ratio_goa_gob = (goa_total / gob_total) if gob_total > 0 else np.nan
    log(
        cfg,
        f"    ROI area={roi_area_um2:.2f} µm² | densities: GOA={goa_density:.4f}/µm², "
        f"GOB={gob_density:.4f}/µm² | ratio GOA:GOB={ratio_goa_gob}",
    )

    per_nucleus_rows: List[Dict] = []
    for lab in range(1, n_nuc + 1):
        nuc_mask = labels == lab
        exp_mask = labels_exp == lab
        area_nuc_um2 = float(nuc_mask.sum() * (cfg.pixel_size_um ** 2))
        area_exp_um2 = float(exp_mask.sum() * (cfg.pixel_size_um ** 2))
        area_ring_um2 = float((exp_mask.sum() - nuc_mask.sum()) * (cfg.pixel_size_um ** 2))

        gob_nuc = int(np.sum(gob_lab_nuc == lab))
        gob_exp = int(np.sum(gob_lab_exp == lab))
        gob_ring = int(gob_exp - gob_nuc)

        goa_nuc = int(np.sum(goa_lab_nuc == lab))
        goa_exp = int(np.sum(goa_lab_exp == lab))
        goa_ring = int(goa_exp - goa_nuc)

        row = {
            "animal": animal,
            "region": region,
            "roi": roi_name,
            "nucleus_id": lab,
            "area_nucleus_um2": area_nuc_um2,
            "area_expanded_um2": area_exp_um2,
            "area_ring_um2": area_ring_um2,
            "GoB_nucleus": gob_nuc,
            "GoB_expanded": gob_exp,
            "GoB_ring": gob_ring,
            "GoA_nucleus": goa_nuc,
            "GoA_expanded": goa_exp,
            "GoA_ring": goa_ring,
            "GoA_bin": count_to_bin(goa_exp),
            "GoB_bin": count_to_bin(gob_exp),
            "more_GoA_than_GoB": int(goa_exp > gob_exp),
            "more_GoB_than_GoA": int(gob_exp > goa_exp),
            "any_GoA": int(goa_exp > 0),
            "any_GoB": int(gob_exp > 0),
            "no_spots_both": int((goa_exp == 0) and (gob_exp == 0)),
        }
        per_nucleus_rows.append(row)

    # Per‑ROI summary
    goa_exp_counts = np.array([r["GoA_expanded"] for r in per_nucleus_rows], dtype=int)
    gob_exp_counts = np.array([r["GoB_expanded"] for r in per_nucleus_rows], dtype=int)
    n_cells = max(1, len(per_nucleus_rows))

    cells_more_goa = int(np.sum(goa_exp_counts > gob_exp_counts))
    cells_more_gob = int(np.sum(gob_exp_counts > goa_exp_counts))
    cells_no_spots = int(np.sum((goa_exp_counts == 0) & (gob_exp_counts == 0)))
    cells_any_goa = int(np.sum(goa_exp_counts > 0))
    cells_any_gob = int(np.sum(gob_exp_counts > 0))

    bins_goa = summarize_bins(goa_exp_counts)
    bins_gob = summarize_bins(gob_exp_counts)
    hscore_goa = sum(b * c for b, c in bins_goa.items())
    hscore_gob = sum(b * c for b, c in bins_gob.items())

    avg_goa_per_cell = float(np.mean(goa_exp_counts)) if len(goa_exp_counts) > 0 else np.nan
    avg_gob_per_cell = float(np.mean(gob_exp_counts)) if len(gob_exp_counts) > 0 else np.nan
    ratio_avg_goa_gob = (
        (avg_goa_per_cell / avg_gob_per_cell) if avg_gob_per_cell > 0 else np.nan
    )

    log(
        cfg,
        "    per‑ROI: "
        f"n_cells={len(per_nucleus_rows)}, moreGOA={cells_more_goa}, "
        f"moreGOB={cells_more_gob}, noSpots={cells_no_spots}, "
        f"Hs(GOA/GOB)={hscore_goa}/{hscore_gob}, "
        f"avg(GOA/GOB)={avg_goa_per_cell:.3f}/{avg_gob_per_cell:.3f}",
    )

    per_roi_row: Dict[str, float] = {
        "animal": animal,
        "region": region,
        "roi": roi_name,
        "n_nuclei": int(len(per_nucleus_rows)),
        "roi_area_um2": roi_area_um2,
        "GoA_total_spots": goa_total,
        "GoB_total_spots": gob_total,
        "GoA_density_per_um2": goa_density,
        "GoB_density_per_um2": gob_density,
        "GoA_to_GoB_ratio": ratio_goa_gob,
        "GoA_avg_spots_per_cell": avg_goa_per_cell,
        "GoB_avg_spots_per_cell": avg_gob_per_cell,
        "GoA_to_GoB_avg_spot_ratio": ratio_avg_goa_gob,
        "cells_more_GoA": cells_more_goa,
        "cells_more_GoA_prop": cells_more_goa / n_cells,
        "cells_more_GoB": cells_more_gob,
        "cells_more_GoB_prop": cells_more_gob / n_cells,
        "cells_no_spots_both": cells_no_spots,
        "cells_no_spots_both_prop": cells_no_spots / n_cells,
        "cells_any_GoA": cells_any_goa,
        "cells_any_GoA_prop": cells_any_goa / n_cells,
        "cells_any_GoB": cells_any_gob,
        "cells_any_GoB_prop": cells_any_gob / n_cells,
        "GoA_Hscore": hscore_goa,
        "GoA_Hscore_norm": hscore_goa / n_cells,
        "GoB_Hscore": hscore_gob,
        "GoB_Hscore_norm": hscore_gob / n_cells,
    }
    for b in BIN_LABELS:
        per_roi_row[f"GoA_bin{b}_count"] = bins_goa[b]
        per_roi_row[f"GoB_bin{b}_count"] = bins_gob[b]
        per_roi_row[f"GoA_bin{b}_prop"] = bins_goa[b] / n_cells
        per_roi_row[f"GoB_bin{b}_prop"] = bins_gob[b] / n_cells

    log(cfg, f"  < ROI '{roi_name}': done in {time.perf_counter() - t0:.2f}s")
    return per_nucleus_rows, per_roi_row
