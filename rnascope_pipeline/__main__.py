"""Command line entry point."""

from __future__ import annotations

from .config import Config
from .pipeline import run_pipeline


def main() -> None:  # pragma: no cover - simple wrapper
    run_pipeline(Config())


if __name__ == "__main__":
    main()
