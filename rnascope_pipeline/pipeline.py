"""High level RNAscope analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import roifile
import tifffile as tiff

from .analysis import analyze_roi, BIN_LABELS, summarize_bins
from .config import Config
from .image_utils import roi_polygon_to_mask
from .segmentation import create_model
from .utils import ensure_dirs, find_maxima_files, get_single_image, log, read_points_roi


def _load_image(path: Path) -> np.ndarray:
    """Load an image and ensure a HWC layout."""
    img = tiff.imread(str(path))
    if img.ndim != 3:
        raise ValueError(f"{path} must be 3‑D; got shape {img.shape}")
    if img.shape[-1] in (3, 4):  # HWC
        layout = "HWC"
    elif img.shape[0] in (3, 4):  # CHW
        img = np.transpose(img, (1, 2, 0))
        layout = "CHW"
    else:
        raise ValueError(f"Cannot infer channel layout for image with shape {img.shape}")
    return img, layout


def run_pipeline(cfg: Config) -> None:
    """Execute the complete RNAscope analysis pipeline."""
    ensure_dirs(cfg)

    model = create_model()
    maxima_dir = cfg.root / "maxima"
    expand_px = int(round(cfg.expansion_um / cfg.pixel_size_um))
    log(cfg, f"=== RNAscope pipeline start | root={cfg.root.resolve()} | expand={cfg.expansion_um}µm≈{expand_px}px ===")

    per_nucleus_all = []
    per_roi_all = []

    experiments = sorted(p for p in cfg.root.iterdir() if p.is_dir() and p.name.lower().startswith("rat"))
    log(cfg, f"found {len(experiments)} experiment(s): {[p.name for p in experiments]}")

    for exp in experiments:
        animal = exp.name
        log(cfg, f"— Processing {animal} —")
        for region, roi_subdir, keyword in (("hippo", "hippo_rois", "hippo"), ("thal", "thal_rois", "thal")):
            roi_dir = exp / roi_subdir
            if not roi_dir.is_dir():
                log(cfg, f"  [SKIP] {animal}/{region}: no ROI dir '{roi_subdir}'")
                continue
            try:
                img_path = get_single_image(exp, keyword)
            except RuntimeError as e:
                log(cfg, f"  [WARN] {e}")
                continue

            full_img, layout = _load_image(img_path)
            log(cfg, f"  [{animal}/{region}] image='{img_path.name}' shape={full_img.shape} layout={layout}")
            H, W = full_img.shape[:2]

            files = find_maxima_files(maxima_dir, animal, region)
            goa_path = files.get("GOA")
            gob_path = files.get("GOB")
            log(
                cfg,
                f"  [{animal}/{region}] maxima files: GOA={(goa_path.name if goa_path else None)}, "
                f"GOB={(gob_path.name if gob_path else None)}",
            )
            goa_x_all, goa_y_all = read_points_roi(goa_path, cfg.transpose_xy)
            gob_x_all, gob_y_all = read_points_roi(gob_path, cfg.transpose_xy)
            log(cfg, f"  [{animal}/{region}] maxima totals: GOA={len(goa_x_all)}, GOB={len(gob_x_all)}")

            roi_files = sorted(roi_dir.glob("*.roi"))
            if not roi_files:
                log(cfg, f"  [SKIP] No .roi files in {roi_dir}")
                continue
            log(cfg, f"  [{animal}/{region}] {len(roi_files)} ROI(s): {[p.stem for p in roi_files]}")

            for roi_path in roi_files:
                roi = roifile.roiread(str(roi_path))
                roi_name = roi_path.stem
                try:
                    mask = roi_polygon_to_mask(roi, (H, W), transpose_xy=cfg.transpose_xy)
                except ValueError as e:
                    if cfg.skip_empty_roi:
                        log(cfg, f"  [SKIP] ROI '{roi_name}' invalid: {e}")
                        continue
                    raise

                pn_rows, pr_row = analyze_roi(
                    cfg=cfg,
                    animal=animal,
                    region=region,
                    roi_name=roi_name,
                    full_img=full_img,
                    roi_mask=mask,
                    gob_xy=(gob_x_all, gob_y_all),
                    goa_xy=(goa_x_all, goa_y_all),
                    model=model,
                    expand_px=expand_px,
                )
                per_nucleus_all.extend(pn_rows)
                per_roi_all.append(pr_row)
                log(
                    cfg,
                    f"  [{animal}/{region}/{roi_name}] nuclei={pr_row['n_nuclei']} "
                    f"GOA={pr_row['GoA_total_spots']} GOB={pr_row['GoB_total_spots']}",
                )

    # Write per-nucleus and per-ROI tables
    per_nuc_df = pd.DataFrame(per_nucleus_all)
    per_roi_df = pd.DataFrame(per_roi_all)

    per_nuc_csv = cfg.csv_dir / "per_nucleus_counts.csv"
    per_roi_csv = cfg.csv_dir / "per_roi_summary.csv"
    per_nuc_df.to_csv(per_nuc_csv, index=False)
    per_roi_df.to_csv(per_roi_csv, index=False)
    log(cfg, f"wrote per-nucleus CSV → {per_nuc_csv}")
    log(cfg, f"wrote per-ROI CSV → {per_roi_csv}")

    # Overall summary
    overall_rows = []
    if not per_nuc_df.empty:
        total_cells = len(per_nuc_df)
        goa_all = per_nuc_df["GoA_expanded"].to_numpy(int)
        gob_all = per_nuc_df["GoB_expanded"].to_numpy(int)
        avg_goa_all = float(goa_all.mean()) if total_cells > 0 else np.nan
        avg_gob_all = float(gob_all.mean()) if total_cells > 0 else np.nan
        avg_ratio_all = (avg_goa_all / avg_gob_all) if avg_gob_all > 0 else np.nan
        cells_more_goa_all = int(np.sum(goa_all > gob_all))
        cells_more_gob_all = int(np.sum(gob_all > goa_all))
        cells_no_spots_all = int(np.sum((goa_all == 0) & (gob_all == 0)))
        cells_any_goa_all = int(np.sum(goa_all > 0))
        cells_any_gob_all = int(np.sum(gob_all > 0))
        bins_goa_all = summarize_bins(goa_all)
        bins_gob_all = summarize_bins(gob_all)
        hscore_goa_all = sum(b * c for b, c in bins_goa_all.items())
        hscore_gob_all = sum(b * c for b, c in bins_gob_all.items())
        overall = {
            "total_cells": total_cells,
            "cells_more_GoA": cells_more_goa_all,
            "cells_more_GoA_prop": cells_more_goa_all / total_cells,
            "cells_more_GoB": cells_more_gob_all,
            "cells_more_GoB_prop": cells_more_gob_all / total_cells,
            "cells_no_spots_both": cells_no_spots_all,
            "cells_no_spots_both_prop": cells_no_spots_all / total_cells,
            "cells_any_GoA": cells_any_goa_all,
            "cells_any_GoA_prop": cells_any_goa_all / total_cells,
            "cells_any_GoB": cells_any_gob_all,
            "cells_any_GoB_prop": cells_any_gob_all / total_cells,
            "GoA_avg_spots_per_cell": avg_goa_all,
            "GoB_avg_spots_per_cell": avg_gob_all,
            "GoA_to_GoB_avg_spot_ratio": avg_ratio_all,
            "GoA_Hscore": hscore_goa_all,
            "GoA_Hscore_norm": hscore_goa_all / total_cells,
            "GoB_Hscore": hscore_gob_all,
            "GoB_Hscore_norm": hscore_gob_all / total_cells,
        }
        for b in BIN_LABELS:
            overall[f"GoA_bin{b}_count"] = bins_goa_all[b]
            overall[f"GoB_bin{b}_count"] = bins_gob_all[b]
            overall[f"GoA_bin{b}_prop"] = bins_goa_all[b] / total_cells
            overall[f"GoB_bin{b}_prop"] = bins_gob_all[b] / total_cells
        overall_rows.append(overall)

    overall_df = pd.DataFrame(overall_rows)
    overall_csv = cfg.csv_dir / "overall_metrics.csv"
    overall_df.to_csv(overall_csv, index=False)
    log(cfg, f"wrote overall CSV → {overall_csv}")

    print(f"Saved per‑nucleus → {per_nuc_csv}")
    print(f"Saved per‑ROI → {per_roi_csv}")
    print(f"Saved overall → {overall_csv}")
    print(
        f"Cutouts in {cfg.cutouts_dir} | Overlays in {cfg.qc_overlays_dir} | Masks in {cfg.masks_dir}"
    )


def main() -> None:  # pragma: no cover - entry point
    run_pipeline(Config())


if __name__ == "__main__":  # pragma: no cover
    main()
