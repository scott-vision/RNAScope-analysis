"""Example launcher for the Tkinter freehand ROI drawer."""

from __future__ import annotations

from pathlib import Path

from rnascope_pipeline.roi_drawer import launch_roi_drawer


def main() -> None:
    root = Path(__file__).resolve().parent
    image_path = root / "Max-Projections" / "Rat1" / "20250804_rat1_hippo_63x.tif"
    output_roi = root / "Max-Projections" / "Rat1" / "hippo_rois" / "example_freehand.roi"
    launch_roi_drawer(image_path=image_path, output_roi=output_roi, roi_name="example_freehand")


if __name__ == "__main__":
    main()
