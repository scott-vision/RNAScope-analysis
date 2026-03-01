#!/usr/bin/env python
"""Generate center-patch QC visualizations for RNAscope Max-Projections data.

For each experiment folder under Max-Projections (e.g. Rat1, Mouse0), this script:
1) Finds hippo/thal region images and polygon ROIs.
2) Runs nuclei segmentation (Cellpose) on masked ROI cutouts (with on-disk cache).
3) Extracts a small patch around the ROI center.
4) Saves a side-by-side PNG: original composite vs segmentation+maxima overlay.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import roifile
import tifffile as tiff
from skimage.io import imsave
from skimage.segmentation import find_boundaries

from rnascope_pipeline.image_utils import draw_crosses_inplace, roi_polygon_to_mask, tight_crop_nd, to_uint8_vis
from rnascope_pipeline.segmentation import create_model, segment_nuclei
from rnascope_pipeline.utils import read_points_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center-patch QC visualization for RNAscope data")
    parser.add_argument("--root", type=Path, default=Path("Max-Projections"), help="Root data directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_center_qc"),
        help="Output directory for PNGs and cached masks",
    )
    parser.add_argument("--patch-size", type=int, default=768, help="Patch size in pixels (square)")
    parser.add_argument("--dapi-index", type=int, default=0)
    parser.add_argument("--gob-index", type=int, default=1)
    parser.add_argument("--goa-index", type=int, default=2)
    parser.add_argument("--transpose-xy", dest="transpose_xy", action="store_true", default=True)
    parser.add_argument("--no-transpose-xy", dest="transpose_xy", action="store_false")
    parser.add_argument("--max-rois-per-region", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--animals",
        nargs="+",
        default=None,
        help="Optional list of experiment folder names to process (e.g. Rat1 Mouse0)",
    )
    parser.add_argument("--load-saved-masks", action="store_true", default=True)
    parser.add_argument("--no-load-saved-masks", dest="load_saved_masks", action="store_false")
    return parser.parse_args()


def _infer_layout(img: np.ndarray) -> Tuple[str, int, int]:
    if img.ndim != 3:
        raise ValueError(f"Expected 3-D image, got shape={img.shape}")
    if img.shape[-1] in (3, 4):
        return "HWC", int(img.shape[0]), int(img.shape[1])
    if img.shape[0] in (3, 4):
        return "CHW", int(img.shape[1]), int(img.shape[2])
    raise ValueError(f"Cannot infer channel layout from shape={img.shape}")


def _find_image(exp_dir: Path, region: str) -> Path | None:
    hits = [
        p
        for p in exp_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and region in p.name.lower()
    ]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"[WARN] {exp_dir.name}/{region}: expected one image, found {len(hits)}; skipping region")
        return None
    return None


def _find_maxima_files(exp_dir: Path, animal: str, region: str) -> Dict[str, Path | None]:
    animal_l = animal.lower()
    region_l = region.lower()
    out: Dict[str, list[Path]] = {"GOA": [], "GOB": []}

    for p in exp_dir.rglob("*.roi"):
        n = p.name.lower()
        if "maxima" not in n:
            continue
        if animal_l not in n or region_l not in n:
            continue
        if "goa" in n:
            out["GOA"].append(p)
        elif "gob" in n:
            out["GOB"].append(p)

    return {
        "GOA": sorted(out["GOA"])[0] if out["GOA"] else None,
        "GOB": sorted(out["GOB"])[0] if out["GOB"] else None,
    }


def _in_roi_and_local(
    x_all: np.ndarray,
    y_all: np.ndarray,
    roi_mask: np.ndarray,
    x0: int,
    y0: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x_all) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    h, w = roi_mask.shape
    x_all = np.asarray(x_all)
    y_all = np.asarray(y_all)
    valid = (x_all >= 0) & (x_all < w) & (y_all >= 0) & (y_all < h)
    x_all = x_all[valid]
    y_all = y_all[valid]
    inside = roi_mask[y_all, x_all]
    return x_all[inside] - x0, y_all[inside] - y0


def _center_on_mask(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    cy = ys.mean()
    cx = xs.mean()
    idx = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    return int(ys[idx]), int(xs[idx])


def _crop_patch(arr: np.ndarray, cy: int, cx: int, size: int) -> Tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    size = max(32, int(size))

    y0 = max(0, cy - size // 2)
    x0 = max(0, cx - size // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)

    if y1 - y0 < size:
        y0 = max(0, y1 - size)
    if x1 - x0 < size:
        x0 = max(0, x1 - size)

    return arr[y0:y1, x0:x1], y0, x0


def _overlay_points(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, color: tuple[int, int, int]) -> None:
    draw_crosses_inplace(rgb, xs, ys, color=color, size=2)


def _panel_original(cutout: np.ndarray, dapi_idx: int, gob_idx: int, goa_idx: int) -> np.ndarray:
    dapi = to_uint8_vis(cutout[..., dapi_idx])
    gob = to_uint8_vis(cutout[..., gob_idx])
    goa = to_uint8_vis(cutout[..., goa_idx])
    return np.stack([goa, gob, dapi], axis=-1)


def _panel_segmented(labels: np.ndarray, dapi: np.ndarray) -> np.ndarray:
    dapi8 = to_uint8_vis(dapi)
    rgb = np.stack([dapi8, dapi8, dapi8], axis=-1)
    boundaries = find_boundaries(labels, mode="outer")
    rgb[boundaries] = (0, 255, 0)
    return rgb


def _to_hwc(cutout: np.ndarray, layout: str) -> np.ndarray:
    if layout == "HWC":
        return np.asarray(cutout)
    return np.moveaxis(np.asarray(cutout), 0, -1)


def _iter_experiments(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p.name.lower().startswith("rat") or p.name.lower().startswith("mouse")))


def main() -> None:
    args = parse_args()

    out_qc = args.out / "qc_panels"
    out_masks = args.out / "masks"
    out_qc.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    model = create_model()
    selected = {a.lower() for a in args.animals} if args.animals else None

    for exp in _iter_experiments(args.root):
        animal = exp.name
        if selected is not None and animal.lower() not in selected:
            continue
        print(f"== {animal} ==")
        for region in ("hippo", "thal"):
            roi_dir = exp / f"{region}_rois"
            if not roi_dir.is_dir():
                continue

            img_path = _find_image(exp, region)
            if img_path is None:
                print(f"  [SKIP] {region}: no unique top-level image")
                continue

            img = tiff.memmap(str(img_path))
            try:
                layout, h, w = _infer_layout(img)
            except ValueError as exc:
                print(f"  [SKIP] {region}: {exc}")
                continue

            maxima = _find_maxima_files(exp, animal, region)
            goa_x_all, goa_y_all = read_points_roi(maxima["GOA"], transpose_xy=args.transpose_xy)
            gob_x_all, gob_y_all = read_points_roi(maxima["GOB"], transpose_xy=args.transpose_xy)
            print(
                f"  [{region}] image={img_path.name} layout={layout} maxima GOA={len(goa_x_all)} GOB={len(gob_x_all)}"
            )

            roi_files = sorted(roi_dir.glob("*.roi"))
            if args.max_rois_per_region > 0:
                roi_files = roi_files[: args.max_rois_per_region]

            for roi_path in roi_files:
                roi_name = roi_path.stem
                try:
                    roi = roifile.roiread(str(roi_path))
                    roi_mask = roi_polygon_to_mask(roi, (h, w), transpose_xy=args.transpose_xy)
                except Exception as exc:
                    print(f"    [SKIP] {roi_name}: invalid ROI ({exc})")
                    continue

                cutout_raw, (y0, y1, x0, x1) = tight_crop_nd(img, roi_mask)
                cutout = _to_hwc(cutout_raw, layout)
                mask_c = roi_mask[y0:y1, x0:x1]
                cutout_masked = cutout * mask_c[..., None]

                goa_x, goa_y = _in_roi_and_local(goa_x_all, goa_y_all, roi_mask, x0, y0)
                gob_x, gob_y = _in_roi_and_local(gob_x_all, gob_y_all, roi_mask, x0, y0)

                dapi = cutout_masked[..., args.dapi_index]
                mask_path = out_masks / f"{animal}_{region}_{roi_name}_labels.tif"
                if args.load_saved_masks and mask_path.exists():
                    labels = tiff.imread(str(mask_path)).astype(np.int32)
                else:
                    labels = segment_nuclei(dapi, model)
                    labels[~mask_c] = 0
                    tiff.imwrite(str(mask_path), labels.astype(np.uint16))

                cy, cx = _center_on_mask(mask_c)

                orig_panel_full = _panel_original(cutout_masked, args.dapi_index, args.gob_index, args.goa_index)
                seg_panel_full = _panel_segmented(labels, dapi)

                orig_patch, py0, px0 = _crop_patch(orig_panel_full, cy, cx, args.patch_size)
                seg_patch, _, _ = _crop_patch(seg_panel_full, cy, cx, args.patch_size)

                goa_px = goa_x - px0
                goa_py = goa_y - py0
                gob_px = gob_x - px0
                gob_py = gob_y - py0

                ph, pw = seg_patch.shape[:2]
                goa_keep = (goa_px >= 0) & (goa_px < pw) & (goa_py >= 0) & (goa_py < ph)
                gob_keep = (gob_px >= 0) & (gob_px < pw) & (gob_py >= 0) & (gob_py < ph)

                _overlay_points(seg_patch, goa_px[goa_keep], goa_py[goa_keep], color=(255, 0, 255))
                _overlay_points(seg_patch, gob_px[gob_keep], gob_py[gob_keep], color=(0, 255, 255))

                sep = np.full((orig_patch.shape[0], 8, 3), 255, dtype=np.uint8)
                panel = np.concatenate([orig_patch, sep, seg_patch], axis=1)

                out_name = f"{animal}_{region}_{roi_name}_center_qc.png"
                imsave(str(out_qc / out_name), panel)
                print(f"    saved {out_name}")

    print(f"Done. QC panels: {out_qc}")
    print("Panel format: left=original composite (R=GoA, G=GoB, B=DAPI), right=Cellpose boundaries + maxima (GOA=magenta, GOB=cyan)")


if __name__ == "__main__":
    main()
