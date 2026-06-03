"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RNAscope analysis pipeline.")
    parser.add_argument("--root", type=Path, default=None, help="Project root containing Rat* folders")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory root")
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=None,
        help="Optional fine-tuned Cellpose model path",
    )
    parser.add_argument(
        "--segmentation-backend",
        default=None,
        help="Output/cache tag for segmentation results",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Model-input patch size for tiled inference, e.g. 256",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Source-to-model downsample factor for tiled inference, e.g. 2",
    )
    species = parser.add_mutually_exclusive_group()
    species.add_argument("--mouse-only", action="store_true", help="Process only Mouse* experiments")
    species.add_argument("--rat-only", action="store_true", help="Process only Rat* experiments")
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment folder names to process, e.g. Mouse0,Rat2",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - simple wrapper
    args = _parse_args()
    cfg = Config()
    if args.root is not None:
        cfg.root = args.root
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.segmentation_model is not None:
        cfg.segmentation_model = args.segmentation_model
    if args.segmentation_backend is not None:
        cfg.segmentation_backend = args.segmentation_backend
    if args.patch_size is not None:
        if args.patch_size <= 0:
            raise ValueError("--patch-size must be > 0")
        cfg.inference_patch_size = args.patch_size
    if args.downsample <= 0:
        raise ValueError("--downsample must be > 0")
    cfg.inference_downsample = args.downsample
    if args.mouse_only:
        cfg.species_filter = "mouse"
    elif args.rat_only:
        cfg.species_filter = "rat"
    if args.experiments:
        cfg.experiments = tuple(
            part.strip() for part in args.experiments.split(",") if part.strip()
        )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
