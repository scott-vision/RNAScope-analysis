"""RNAscope analysis pipeline package."""

from .config import Config

__all__ = ["Config", "run_pipeline"]


def run_pipeline(cfg: Config) -> None:
    """Lazily import and execute the full pipeline."""
    from .pipeline import run_pipeline as _run_pipeline

    _run_pipeline(cfg)
