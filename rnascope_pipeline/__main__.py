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
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
