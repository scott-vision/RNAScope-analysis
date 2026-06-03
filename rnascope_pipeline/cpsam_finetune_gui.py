
"""Tkinter interface for Cellpose-SAM fine-tuning on RNAscope patches."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import queue
import random
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.transform import resize
import tifffile as tiff

from .dataset_store import DatasetStore


VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SESSION_PATHS_FILE = Path(__file__).resolve().parents[1] / "cpsam_last_paths.txt"
BEST_VAL_SAVE_EVERY = 5


@dataclass
class PatchItem:
    patch_id: str
    image_path: Path
    mask_path: Path
    source_id: str = ""
    annotation_status: str = "auto"
    include: bool = True
    split: str = "unassigned"


class _TrainingLogWriter:
    def __init__(self, emit, log_fh) -> None:
        self.emit = emit
        self.log_fh = log_fh
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.emit(line.rstrip())
            self.log_fh.write(line.rstrip() + "\n")
            self.log_fh.flush()
        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self.emit(self.buffer.rstrip())
            self.log_fh.write(self.buffer.rstrip() + "\n")
            self.log_fh.flush()
            self.buffer = ""


def _percentile_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        g = _percentile_to_uint8(arr)
        return np.stack([g, g, g], axis=-1)

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            rgb = np.transpose(arr[:3, ...], (1, 2, 0))
        else:
            g = _percentile_to_uint8(np.max(arr, axis=0))
            return np.stack([g, g, g], axis=-1)

        if rgb.dtype == np.uint8:
            return rgb
        out = np.zeros(rgb.shape, dtype=np.uint8)
        for i in range(3):
            out[..., i] = _percentile_to_uint8(rgb[..., i])
        return out

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _scale_to_uint8(arr: np.ndarray, low_pct: float, high_pct: float, gamma: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    norm = np.power(norm, gamma)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _to_rgb_with_contrast(
    image: np.ndarray,
    low_pct: float,
    high_pct: float,
    gamma: float,
    show_dapi_gob_goa: tuple[bool, bool, bool] = (True, True, True),
    channel_indices: tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        g = _scale_to_uint8(arr, low_pct=low_pct, high_pct=high_pct, gamma=gamma)
        show_dapi, _show_gob, _show_goa = show_dapi_gob_goa
        out = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
        if show_dapi:
            out[..., 2] = g
        return out

    if arr.ndim == 3:
        dapi_idx, gob_idx, goa_idx = channel_indices
        max_idx = max(dapi_idx, gob_idx, goa_idx)

        # Prefer explicit channel-axis detection for microscopy layouts.
        if arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4) and arr.shape[-1] > max_idx:
            dapi = arr[..., dapi_idx]
            gob = arr[..., gob_idx]
            goa = arr[..., goa_idx]
        elif arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4) and arr.shape[0] > max_idx:
            dapi = arr[dapi_idx, ...]
            gob = arr[gob_idx, ...]
            goa = arr[goa_idx, ...]
        elif arr.shape[-1] > max_idx and arr.shape[-1] < min(arr.shape[0], arr.shape[1]):
            dapi = arr[..., dapi_idx]
            gob = arr[..., gob_idx]
            goa = arr[..., goa_idx]
        elif arr.shape[0] > max_idx and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            dapi = arr[dapi_idx, ...]
            gob = arr[gob_idx, ...]
            goa = arr[goa_idx, ...]
        elif arr.shape[0] > max_idx:
            # Ambiguous 3-D stack: prefer first axis as channels/planes.
            dapi = arr[dapi_idx, ...]
            gob = arr[gob_idx, ...]
            goa = arr[goa_idx, ...]
        elif arr.shape[-1] > max_idx:
            dapi = arr[..., dapi_idx]
            gob = arr[..., gob_idx]
            goa = arr[..., goa_idx]
        else:
            raise ValueError(
                f"Cannot map channels for shape {arr.shape} with indices {channel_indices}. "
                "Set D/G/A indices to match your data layout."
            )

        # Display mapping matches QC panels:
        #   Red=GoA(channel 2), Green=GoB(channel 1), Blue=DAPI(channel 0)
        show_dapi, show_gob, show_goa = show_dapi_gob_goa
        out = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype=np.uint8)
        if show_goa:
            out[..., 0] = _scale_to_uint8(goa, low_pct=low_pct, high_pct=high_pct, gamma=gamma)
        if show_gob:
            out[..., 1] = _scale_to_uint8(gob, low_pct=low_pct, high_pct=high_pct, gamma=gamma)
        if show_dapi:
            out[..., 2] = _scale_to_uint8(dapi, low_pct=low_pct, high_pct=high_pct, gamma=gamma)
        return out

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return np.mean(arr[..., :3], axis=-1)
        if arr.shape[0] in (3, 4):
            return np.mean(arr[:3, ...], axis=0)
        return np.max(arr, axis=0)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _extract_nuclei_channel(image: np.ndarray, channel_index: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and channel_index < arr.shape[-1]:
            return arr[..., channel_index]
        if arr.shape[0] in (3, 4) and channel_index < arr.shape[0]:
            return arr[channel_index, ...]
        if channel_index < arr.shape[-1]:
            return arr[..., channel_index]
        if channel_index < arr.shape[0]:
            return arr[channel_index, ...]
    raise ValueError(f"Cannot extract nuclei channel {channel_index} from shape {arr.shape}")


def _rgb_to_ppm_bytes(rgb: np.ndarray) -> bytes:
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError("Expected RGB image")
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + rgb.tobytes(order="C")


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _sample_patch(img: np.ndarray, patch_size: int, rng: random.Random) -> np.ndarray:
    patch, _x0, _y0, _w, _h = _sample_patch_with_coords(img, patch_size=patch_size, rng=rng)
    return patch


def _resize_patch_to_size(patch: np.ndarray, patch_size: int) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.ndim == 2:
        target_shape = (patch_size, patch_size)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        target_shape = (arr.shape[0], patch_size, patch_size)
    elif arr.ndim == 3:
        target_shape = (patch_size, patch_size, arr.shape[-1])
    else:
        raise ValueError(f"Unsupported patch shape for resizing: {arr.shape}")

    if arr.shape == target_shape:
        return np.asarray(arr)

    resized = resize(
        arr,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        resized = np.clip(np.rint(resized), info.min, info.max).astype(arr.dtype)
    else:
        resized = resized.astype(arr.dtype, copy=False)
    return np.asarray(resized)


def _sample_patch_with_coords(
    img: np.ndarray, patch_size: int, rng: random.Random, downsample_factor: int = 1
) -> tuple[np.ndarray, int, int, int, int]:
    arr = np.asarray(img)
    if arr.ndim < 2:
        raise ValueError(f"Unsupported image shape for patching: {arr.shape}")

    if arr.ndim == 2:
        h, w = arr.shape
        layout = "HW"
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        h, w = arr.shape[1], arr.shape[2]
        layout = "CHW"
    else:
        h, w = arr.shape[0], arr.shape[1]
        layout = "HWC"

    source_size = patch_size * max(1, int(downsample_factor))
    if h < source_size or w < source_size:
        raise ValueError(f"Image too small for {source_size}x{source_size}: {arr.shape}")
    y0 = rng.randint(0, h - source_size)
    x0 = rng.randint(0, w - source_size)

    if layout == "HW":
        patch = arr[y0 : y0 + source_size, x0 : x0 + source_size]
    elif layout == "CHW":
        patch = arr[:, y0 : y0 + source_size, x0 : x0 + source_size]
    else:
        patch = arr[y0 : y0 + source_size, x0 : x0 + source_size, ...]
    patch = _resize_patch_to_size(np.asarray(patch), patch_size)
    return np.asarray(patch), x0, y0, source_size, source_size


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class CpsamFineTuneApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("RNAscope Cellpose-SAM Fine-Tuning")
        self.root.geometry("1800x1050")
        self.root.minsize(1400, 860)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.training_cancel_requested = False
        self.training_active = False

        self.display_size = 512
        self.current_display_w = self.display_size
        self.current_display_h = self.display_size
        self.current_display_x = 0
        self.current_display_y = 0

        self.patch_items: list[PatchItem] = []
        self.current_index = -1
        self.current_patch_id: str = ""
        self.current_source_img: np.ndarray | None = None
        self.current_img: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.current_photo: tk.PhotoImage | None = None
        self.roi_draw_items: list[int] = []

        self.mode_var = tk.StringVar(value="point")
        self.freehand_points: list[tuple[float, float]] = []
        self.freehand_line_item: int | None = None

        self.source_dir_var = tk.StringVar()
        self.workspace_dir_var = tk.StringVar()
        self.source_scope_var = tk.StringVar(value="current")
        self.patch_size_var = tk.StringVar(value="256")
        self.downsample_factor_var = tk.StringVar(value="2")
        self.train_count_var = tk.StringVar(value="50")
        self.gpu_var = tk.BooleanVar(value=True)
        self.model_name_out_var = tk.StringVar(value="cpsam_rnascope")
        self.ultra_sam2_ckpt_var = tk.StringVar(value="sam2_b.pt")
        self.trained_model_path_var = tk.StringVar()

        self.learning_rate_var = tk.StringVar(value="0.00001")
        self.weight_decay_var = tk.StringVar(value="0.1")
        self.n_epochs_var = tk.StringVar(value="100")
        self.train_batch_size_var = tk.StringVar(value="1")
        self.min_train_masks_var = tk.StringVar(value="1")
        self.train_verbose_var = tk.BooleanVar(value=True)
        self.train_best_val_var = tk.BooleanVar(value=True)
        self.contrast_low_var = tk.StringVar(value="1")
        self.contrast_high_var = tk.StringVar(value="99")
        self.contrast_gamma_var = tk.StringVar(value="1.0")
        self.show_dapi_var = tk.BooleanVar(value=True)
        self.show_gob_var = tk.BooleanVar(value=True)
        self.show_goa_var = tk.BooleanVar(value=True)
        self.dapi_index_var = tk.StringVar(value="0")
        self.gob_index_var = tk.StringVar(value="1")
        self.goa_index_var = tk.StringVar(value="2")
        self.patch_status_var = tk.StringVar(value="auto")
        self.patch_include_var = tk.BooleanVar(value=True)
        self.patch_split_var = tk.StringVar(value="unassigned")
        self.snapshot_name_var = tk.StringVar(value="snapshot")
        self.snapshot_choice_var = tk.StringVar(value="")
        self.snapshot_display_var = tk.StringVar(value="")

        self.train_cmd_var = tk.StringVar(value="")
        self.store: DatasetStore | None = None
        self.snapshot_choices: list[str] = []
        self.snapshot_display_to_id: dict[str, str] = {}
        self.snapshot_id_to_display: dict[str, str] = {}
        self.eval_preview_items: list[tuple[Path, Path, Path | None]] = []
        self.eval_preview_index: int = -1
        self.eval_preview_status_var = tk.StringVar(value="Eval results: 0")
        self.current_eval_img_path: Path | None = None
        self.current_eval_pred_mask_path: Path | None = None
        self.eval_display_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.run_id_active: str | None = None
        self.logs_visible = False

        self._load_last_paths()
        self._build_layout()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._drain_logs)
        self._refresh_sources_list()
        self._refresh_snapshot_choices()
        self._refresh_runs_list()
        self.load_train_set()

    @property
    def workspace_dir(self) -> Path:
        return Path(self.workspace_dir_var.get()).expanduser().resolve()

    @property
    def train_dir(self) -> Path:
        return self.workspace_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.workspace_dir / "test"

    @property
    def eval_results_dir(self) -> Path:
        return self.workspace_dir / "eval_results"

    def _get_store(self) -> DatasetStore:
        if not self.workspace_dir_var.get().strip():
            self.workspace_dir_var.set(str(Path(".").resolve()))
        if self.store is None or self.store.workspace != self.workspace_dir:
            self.store = DatasetStore(self.workspace_dir)
        return self.store

    def _build_layout(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10, 6))

        tk.Label(top, text="Source Image File").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.source_dir_var, width=90).grid(row=0, column=1, padx=4)
        tk.Button(top, text="Browse", command=self._browse_source_file).grid(row=0, column=2, padx=2)

        tk.Label(top, text="Workspace Folder").grid(row=1, column=0, sticky="w")
        tk.Entry(top, textvariable=self.workspace_dir_var, width=90).grid(row=1, column=1, padx=4)
        tk.Button(top, text="Browse", command=self._browse_workspace_dir).grid(row=1, column=2, padx=2)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(controls, text="Patch size").grid(row=0, column=0, sticky="w")
        tk.Entry(controls, textvariable=self.patch_size_var, width=8).grid(row=0, column=1, sticky="w")
        tk.Label(controls, text="Downsample factor").grid(row=0, column=2, sticky="w", padx=(10, 0))
        tk.Entry(controls, textvariable=self.downsample_factor_var, width=8).grid(row=0, column=3, sticky="w")
        tk.Label(controls, text="Train patches").grid(row=0, column=4, sticky="w", padx=(10, 0))
        tk.Entry(controls, textvariable=self.train_count_var, width=8).grid(row=0, column=5, sticky="w")
        tk.Checkbutton(controls, text="Use GPU", variable=self.gpu_var).grid(
            row=0, column=6, padx=(10, 0), sticky="w"
        )
        tk.Label(controls, text="Sampling scope").grid(row=0, column=7, sticky="w", padx=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.source_scope_var,
            values=["current", "selected", "all_active"],
            width=12,
            state="readonly",
        ).grid(row=0, column=8, sticky="w")

        tk.Button(
            controls, text="1) Generate Train Patches", command=self.generate_train_patches
        ).grid(row=1, column=0, columnspan=2, pady=(6, 0), sticky="w")
        tk.Button(
            controls, text="2) Auto-Label with Cellpose-SAM", command=self.autolabel_train_patches
        ).grid(row=1, column=2, columnspan=2, pady=(6, 0), sticky="w")
        tk.Button(controls, text="3) Load Train Set", command=self.load_train_set).grid(
            row=1, column=4, columnspan=2, pady=(6, 0), sticky="w"
        )
        tk.Button(controls, text="Save Current Mask", command=self.save_current_mask).grid(
            row=1, column=6, pady=(6, 0), sticky="w"
        )

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        left = tk.Frame(body)
        left.pack(side="left", fill="y")
        src_box = tk.LabelFrame(left, text="Sources")
        src_box.pack(fill="x", pady=(0, 6))
        self.sources_listbox = tk.Listbox(
            src_box, width=36, height=6, selectmode="extended", exportselection=False
        )
        src_scroll_x = tk.Scrollbar(src_box, orient="horizontal", command=self.sources_listbox.xview)
        self.sources_listbox.configure(xscrollcommand=src_scroll_x.set)
        self.sources_listbox.pack(fill="x", padx=4, pady=(4, 0))
        src_scroll_x.pack(fill="x", padx=4, pady=(0, 4))
        src_btns = tk.Frame(src_box)
        src_btns.pack(fill="x", padx=4, pady=(0, 4))
        tk.Button(src_btns, text="Add Current", command=self.add_current_source).pack(side="left")
        tk.Button(src_btns, text="Remove", command=self.remove_selected_sources).pack(
            side="left", padx=(4, 0)
        )
        tk.Button(src_btns, text="Refresh", command=self._refresh_sources_list).pack(
            side="left", padx=(4, 0)
        )

        tk.Label(left, text="Train Patches").pack(anchor="w")
        self.patch_listbox = tk.Listbox(
            left, width=36, height=28, selectmode="extended", exportselection=False
        )
        self.patch_listbox.pack(fill="y", expand=False)
        patch_progress = tk.Frame(left)
        patch_progress.pack(fill="x", pady=(4, 2))
        self.patch_progress_var = tk.DoubleVar(value=0.0)
        self.patch_progress_label_var = tk.StringVar(value="Ready")
        self.patch_progress_bar = ttk.Progressbar(
            patch_progress,
            variable=self.patch_progress_var,
            maximum=100,
            mode="determinate",
            length=240,
        )
        self.patch_progress_bar.pack(fill="x")
        tk.Label(patch_progress, textvariable=self.patch_progress_label_var).pack(anchor="w")

        nav = tk.Frame(left)
        nav.pack(fill="x", pady=(4, 8))
        tk.Button(nav, text="Prev", command=self.prev_patch).pack(side="left", padx=(0, 4))
        tk.Button(nav, text="Next", command=self.next_patch).pack(side="left", padx=(0, 4))
        tk.Button(nav, text="Delete Selected", command=self.delete_selected_patches).pack(side="left", padx=(0, 4))
        self.patch_status_label = tk.Label(nav, text="No patch")
        self.patch_status_label.pack(side="left")

        tools = tk.LabelFrame(left, text="Annotation Tools")
        tools.pack(fill="x", pady=(0, 8))
        tk.Radiobutton(
            tools, text="Add ROI from point (SAM2)", variable=self.mode_var, value="point"
        ).pack(anchor="w")
        tk.Radiobutton(
            tools, text="Draw freehand ROI", variable=self.mode_var, value="freehand"
        ).pack(anchor="w")
        tk.Label(tools, text="Tip: click any red X to delete that ROI").pack(anchor="w", pady=(4, 0))
        tk.Label(tools, text="SAM2 checkpoint").pack(anchor="w", pady=(6, 0))
        tk.Entry(tools, textvariable=self.ultra_sam2_ckpt_var, width=28).pack(anchor="w")

        patch_meta = tk.LabelFrame(left, text="Patch Metadata")
        patch_meta.pack(fill="x", pady=(0, 6))
        tk.Label(patch_meta, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            patch_meta,
            textvariable=self.patch_status_var,
            values=["auto", "edited", "approved", "rejected"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w")
        tk.Label(patch_meta, text="Split").grid(row=1, column=0, sticky="w")
        self.patch_split_combo = ttk.Combobox(
            patch_meta,
            textvariable=self.patch_split_var,
            values=["unassigned", "train", "val", "test"],
            state="readonly",
            width=10,
        )
        self.patch_split_combo.grid(row=1, column=1, sticky="w")
        self.patch_split_combo.bind("<<ComboboxSelected>>", self._on_split_selected)
        tk.Checkbutton(patch_meta, text="Include", variable=self.patch_include_var).grid(
            row=2, column=0, sticky="w"
        )
        tk.Button(patch_meta, text="Apply Metadata", command=self.apply_patch_metadata).grid(
            row=2, column=1, sticky="w"
        )

        center = tk.Frame(body)
        center.pack(side="left", fill="both", expand=True, padx=8)
        tk.Label(center, text="Current Patch + ROIs").pack(anchor="w")
        self.canvas = tk.Canvas(
            center,
            width=self.display_size,
            height=self.display_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#999",
            cursor="crosshair",
        )
        self.canvas.pack(anchor="n", pady=(0, 4))
        self.toggle_logs_btn = tk.Button(center, text="Show Logs", command=self.toggle_logs_visibility)
        self.toggle_logs_btn.pack(anchor="w", pady=(6, 2))
        self.log_frame = tk.LabelFrame(center, text="Log")
        self.log_text = tk.Text(self.log_frame, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        right = tk.LabelFrame(body, text="Training / Evaluation")
        right.pack(side="left", fill="both", expand=True)

        row = 0
        contrast = tk.LabelFrame(right, text="View Contrast")
        contrast.grid(row=row, column=0, columnspan=2, sticky="we", pady=(0, 6))
        tk.Label(contrast, text="Low %").grid(row=0, column=0, sticky="w")
        tk.Entry(contrast, textvariable=self.contrast_low_var, width=6).grid(row=0, column=1, sticky="w")
        tk.Label(contrast, text="High %").grid(row=0, column=2, sticky="w", padx=(8, 0))
        tk.Entry(contrast, textvariable=self.contrast_high_var, width=6).grid(row=0, column=3, sticky="w")
        tk.Label(contrast, text="Gamma").grid(row=0, column=4, sticky="w", padx=(8, 0))
        tk.Entry(contrast, textvariable=self.contrast_gamma_var, width=6).grid(row=0, column=5, sticky="w")
        tk.Button(contrast, text="Apply", command=self.apply_contrast_settings).grid(
            row=0, column=6, padx=(8, 0), sticky="w"
        )
        tk.Button(contrast, text="Reset", command=self.reset_contrast_settings).grid(
            row=0, column=7, padx=(4, 0), sticky="w"
        )
        tk.Checkbutton(
            contrast, text="DAPI (Blue)", variable=self.show_dapi_var, command=self.apply_contrast_settings
        ).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(
            contrast, text="GoB (Green)", variable=self.show_gob_var, command=self.apply_contrast_settings
        ).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(
            contrast, text="GoA (Red)", variable=self.show_goa_var, command=self.apply_contrast_settings
        ).grid(row=1, column=2, sticky="w")
        row += 1

        snapshot = tk.LabelFrame(right, text="Dataset Snapshot")
        snapshot.grid(row=row, column=0, columnspan=2, sticky="we", pady=(0, 6))
        tk.Label(snapshot, text="Name").grid(row=0, column=0, sticky="w")
        tk.Entry(snapshot, textvariable=self.snapshot_name_var, width=16).grid(row=0, column=1, sticky="w")
        tk.Button(snapshot, text="Build Snapshot", command=self.build_snapshot).grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )
        tk.Label(snapshot, text="Use snapshot").grid(row=1, column=0, sticky="w")
        self.snapshot_combo = ttk.Combobox(
            snapshot, textvariable=self.snapshot_display_var, values=[], width=48, state="readonly"
        )
        self.snapshot_combo.grid(row=1, column=1, columnspan=2, sticky="w")
        self.snapshot_combo.bind("<<ComboboxSelected>>", self._on_snapshot_selected)
        tk.Button(snapshot, text="Refresh", command=self._refresh_snapshot_choices).grid(
            row=1, column=3, sticky="w", padx=(6, 0)
        )
        tk.Button(snapshot, text="Delete", command=self.delete_snapshot).grid(
            row=1, column=4, sticky="w", padx=(6, 0)
        )
        row += 1

        tk.Label(right, text="Model name out").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.model_name_out_var, width=28).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        tk.Label(right, text="Learning rate").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.learning_rate_var, width=12).grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        tk.Label(right, text="Weight decay").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.weight_decay_var, width=12).grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        tk.Label(right, text="Epochs").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.n_epochs_var, width=12).grid(row=row, column=1, sticky="w")
        row += 1
        tk.Label(right, text="Train batch size").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.train_batch_size_var, width=12).grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        tk.Label(right, text="Min train masks").grid(row=row, column=0, sticky="w")
        tk.Entry(right, textvariable=self.min_train_masks_var, width=12).grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        tk.Checkbutton(right, text="Verbose training logs (--verbose)", variable=self.train_verbose_var).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1
        tk.Checkbutton(
            right,
            text="Pick best val checkpoint (save_each)",
            variable=self.train_best_val_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        tk.Button(right, text="4) Build Train API", command=self.populate_train_command).grid(
            row=row, column=0, sticky="w", pady=(8, 0)
        )
        tk.Button(right, text="5) Run Training", command=self.run_training_command).grid(
            row=row, column=1, sticky="w", pady=(8, 0)
        )
        tk.Button(right, text="Cancel Training", command=self.cancel_training).grid(
            row=row, column=1, sticky="e", pady=(8, 0)
        )
        row += 1

        tk.Label(right, text="API training call").grid(row=row, column=0, sticky="nw", pady=(4, 0))
        tk.Entry(right, textvariable=self.train_cmd_var, width=75).grid(
            row=row, column=1, sticky="w", pady=(4, 0)
        )
        row += 1

        tk.Label(right, text="Trained model path").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(right, textvariable=self.trained_model_path_var, width=55).grid(
            row=row, column=1, sticky="w", pady=(6, 0)
        )
        row += 1
        tk.Button(right, text="Browse Model", command=self._browse_model_file).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        tk.Button(
            right, text="6) Run New Model on Snapshot Test Patches", command=self.run_eval_inference
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1
        eval_preview = tk.Frame(right)
        eval_preview.grid(row=row, column=0, columnspan=2, sticky="we", pady=(4, 0))
        tk.Button(eval_preview, text="Load Eval Results", command=self.load_eval_results).pack(
            side="left"
        )
        tk.Button(eval_preview, text="Prev Eval", command=self.prev_eval_result).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(eval_preview, text="Next Eval", command=self.next_eval_result).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(eval_preview, text="Add Eval To Train Set", command=self.add_current_eval_to_train_set).pack(
            side="left", padx=(6, 0)
        )
        tk.Label(eval_preview, textvariable=self.eval_preview_status_var).pack(side="left", padx=(10, 0))
        row += 1

        eval_list = tk.LabelFrame(right, text="Eval Results")
        eval_list.grid(row=row, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.eval_listbox = tk.Listbox(eval_list, width=70, height=7, exportselection=False)
        self.eval_listbox.pack(fill="x", padx=4, pady=4)
        self.eval_listbox.bind("<<ListboxSelect>>", self._on_eval_select)
        row += 1

        runs = tk.LabelFrame(right, text="Runs")
        runs.grid(row=row, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.runs_listbox = tk.Listbox(runs, width=65, height=6)
        self.runs_listbox.pack(fill="x", padx=4, pady=4)
        tk.Button(runs, text="Refresh Runs", command=self._refresh_runs_list).pack(anchor="w", padx=4, pady=(0, 4))

    def _bind_events(self) -> None:
        self.patch_listbox.bind("<<ListboxSelect>>", self._on_patch_select)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.bind("<KeyPress-s>", self._on_save_shortcut)
        self.root.bind("<KeyPress-S>", self._on_save_shortcut)

    def _on_save_shortcut(self, event: tk.Event) -> str | None:
        widget = event.widget
        widget_class = ""
        try:
            widget_class = str(widget.winfo_class())
        except tk.TclError:
            pass
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}:
            return None
        self.save_current_mask()
        return "break"

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.current_img is not None:
            self._render_current_patch()

    def _log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(200, self._drain_logs)

    def toggle_logs_visibility(self) -> None:
        if self.logs_visible:
            self.log_frame.pack_forget()
            self.toggle_logs_btn.configure(text="Show Logs")
            self.logs_visible = False
        else:
            self.log_frame.pack(fill="both", expand=True, pady=(0, 4))
            self.toggle_logs_btn.configure(text="Hide Logs")
            self.logs_visible = True

    def _browse_source_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select source image file",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if path:
            self.source_dir_var.set(path)
            self._save_last_paths()

    def _browse_workspace_dir(self) -> None:
        folder = filedialog.askdirectory(title="Select workspace folder")
        if folder:
            self.workspace_dir_var.set(folder)
            self._save_last_paths()
            self.store = None
            self._refresh_sources_list()
            self._refresh_snapshot_choices()
            self._refresh_runs_list()

    def _browse_model_file(self) -> None:
        path = filedialog.askopenfilename(title="Select trained model", filetypes=[("All", "*.*")])
        if path:
            self.trained_model_path_var.set(path)

    def _refresh_sources_list(self) -> None:
        if not hasattr(self, "sources_listbox"):
            return
        self.sources_listbox.delete(0, "end")
        try:
            sources = self._get_store().load_sources()
        except Exception:
            return
        for s in sources:
            state = "active" if s.active else "inactive"
            self.sources_listbox.insert("end", f"{s.source_id} | {state} | {s.path}")

    def add_current_source(self) -> None:
        src = self.source_dir_var.get().strip()
        if not src:
            messagebox.showerror("Missing source", "Set Source Image File first.")
            return
        path = Path(src).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            messagebox.showerror("Invalid source", f"Not a valid image file: {path}")
            return
        self._get_store().upsert_source(path)
        self._refresh_sources_list()
        self._log(f"Added source: {path}")

    def remove_selected_sources(self) -> None:
        sel = list(self.sources_listbox.curselection())
        if not sel:
            return
        sources = self._get_store().load_sources()
        for idx in reversed(sel):
            if 0 <= idx < len(sources):
                self._get_store().remove_source(sources[idx].source_id)
        self._refresh_sources_list()
        self._log("Removed selected sources.")

    def _selected_source_ids(self) -> list[str]:
        indices = list(self.sources_listbox.curselection())
        sources = self._get_store().load_sources()
        out: list[str] = []
        for i in indices:
            if 0 <= i < len(sources):
                out.append(sources[i].source_id)
        return out

    def _source_map_for_scope(self) -> dict[str, list[Path]]:
        store = self._get_store()
        scope = self.source_scope_var.get().strip()
        sources = store.load_sources()
        by_id: dict[str, list[Path]] = {}

        if scope == "current":
            cur = self.source_dir_var.get().strip()
            if not cur:
                return {}
            p = Path(cur).expanduser().resolve()
            if not p.is_file() or p.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                return {}
            entry = store.upsert_source(p)
            by_id[entry.source_id] = [Path(entry.path)]
            return by_id

        if scope == "selected":
            allow = set(self._selected_source_ids())
            sources = [s for s in sources if s.source_id in allow]
        elif scope == "all_active":
            sources = [s for s in sources if s.active]

        for s in sources:
            sp = Path(s.path)
            if sp.is_file() and sp.suffix.lower() in VALID_IMAGE_SUFFIXES:
                by_id[s.source_id] = [sp]
            elif sp.is_dir():
                # Backward compatibility for previously saved folder sources.
                imgs = _list_images(sp)
                if imgs:
                    by_id[s.source_id] = imgs
        return by_id

    def _refresh_snapshot_choices(self) -> None:
        if not hasattr(self, "snapshot_combo"):
            return
        snapshots = self._get_store().list_snapshots()
        snap_ids = [s.get("snapshot_id", "") for s in snapshots if s.get("snapshot_id")]
        self.snapshot_display_to_id.clear()
        self.snapshot_id_to_display.clear()
        display_choices: list[str] = []
        for snap in snapshots:
            snapshot_id = str(snap.get("snapshot_id", "")).strip()
            if not snapshot_id:
                continue
            name = str(snap.get("name", "")).strip()
            display = f"{name} [{snapshot_id}]" if name else snapshot_id
            display_choices.append(display)
            self.snapshot_display_to_id[display] = snapshot_id
            self.snapshot_id_to_display[snapshot_id] = display
        self.snapshot_choices = snap_ids
        self.snapshot_combo.configure(values=display_choices)
        current_snapshot_id = self.snapshot_choice_var.get().strip()
        if snap_ids and current_snapshot_id not in snap_ids:
            current_snapshot_id = snap_ids[-1]
        if current_snapshot_id:
            self._set_snapshot_choice(current_snapshot_id)
        if not snap_ids:
            self.snapshot_choice_var.set("")
            self.snapshot_display_var.set("")

    def _set_snapshot_choice(self, snapshot_id: str) -> None:
        sid = str(snapshot_id).strip()
        self.snapshot_choice_var.set(sid)
        self.snapshot_display_var.set(self.snapshot_id_to_display.get(sid, sid))

    def _on_snapshot_selected(self, _event: tk.Event | None = None) -> None:
        display = self.snapshot_display_var.get().strip()
        if not display:
            self.snapshot_choice_var.set("")
            return
        snapshot_id = self.snapshot_display_to_id.get(display, display)
        self.snapshot_choice_var.set(snapshot_id)

    def _refresh_runs_list(self) -> None:
        if not hasattr(self, "runs_listbox"):
            return
        self.runs_listbox.delete(0, "end")
        try:
            runs = self._get_store().list_runs()
        except Exception:
            runs = []
        for r in reversed(runs[-50:]):
            self.runs_listbox.insert(
                "end",
                f"{r.get('run_id','')} | {r.get('status','')} | snapshot={r.get('snapshot_id','')}",
            )

    def _validate_common_inputs(self) -> tuple[Path, Path, int, int] | None:
        if not self.workspace_dir_var.get().strip():
            messagebox.showerror("Missing paths", "Set Workspace Folder.")
            return None
        source_val = self.source_dir_var.get().strip()
        source_dir = Path(source_val).expanduser().resolve() if source_val else Path(".").resolve()
        try:
            patch_size = int(self.patch_size_var.get())
        except ValueError:
            messagebox.showerror("Invalid patch size", "Patch size must be an integer.")
            return None
        if patch_size <= 0:
            messagebox.showerror("Invalid patch size", "Patch size must be > 0.")
            return None
        try:
            downsample_factor = int(self.downsample_factor_var.get())
        except ValueError:
            messagebox.showerror("Invalid downsample factor", "Downsample factor must be an integer.")
            return None
        if downsample_factor <= 0:
            messagebox.showerror("Invalid downsample factor", "Downsample factor must be > 0.")
            return None
        _ensure_dir(self.workspace_dir)
        self._get_store()
        self._save_last_paths()
        return source_dir, self.workspace_dir, patch_size, downsample_factor

    def _load_last_paths(self) -> None:
        if not SESSION_PATHS_FILE.exists():
            return
        try:
            lines = SESSION_PATHS_FILE.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        data: dict[str, str] = {}
        for line in lines:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip().lower()] = v.strip()
        source = data.get("source")
        workspace = data.get("workspace")
        if source:
            self.source_dir_var.set(source)
        if workspace:
            self.workspace_dir_var.set(workspace)

    def _save_last_paths(self) -> None:
        source = self.source_dir_var.get().strip()
        workspace = self.workspace_dir_var.get().strip()
        payload = f"source={source}\nworkspace={workspace}\n"
        try:
            SESSION_PATHS_FILE.write_text(payload, encoding="utf-8")
        except OSError:
            # Best-effort persistence; ignore IO errors and continue.
            return

    def _on_close(self) -> None:
        self._save_last_paths()
        self.root.destroy()

    def _load_patch_pairs(self) -> list[PatchItem]:
        rows = self._get_store().read_manifest()
        items: list[PatchItem] = []
        for r in rows:
            try:
                img_path = Path(r.get("img_path", ""))
                mask_path = Path(r.get("mask_path", ""))
            except TypeError:
                continue
            if not img_path.exists() or not mask_path.exists():
                continue
            items.append(
                PatchItem(
                    patch_id=r.get("patch_id", ""),
                    image_path=img_path,
                    mask_path=mask_path,
                    source_id=r.get("source_id", ""),
                    annotation_status=r.get("annotation_status", "auto"),
                    include=r.get("include", "true").lower() == "true",
                    split=r.get("split", "unassigned"),
                )
            )
        items.sort(key=lambda x: x.patch_id)
        return items

    def _set_patch_progress(self, value: float, label: str) -> None:
        if not hasattr(self, "patch_progress_var"):
            return
        self.patch_progress_var.set(max(0.0, min(100.0, float(value))))
        self.patch_progress_label_var.set(label)
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _set_patch_progress_threadsafe(self, value: float, label: str) -> None:
        try:
            self.root.after(0, lambda: self._set_patch_progress(value, label))
        except tk.TclError:
            pass

    def _count_labeled_pairs(self, folder: Path) -> int:
        if not folder.exists():
            return 0
        count = 0
        for img_path in folder.glob("*_img.tif"):
            mask_path = folder / img_path.name.replace("_img.tif", "_masks.tif")
            if mask_path.exists():
                count += 1
        return count

    def generate_train_patches(self) -> None:
        validated = self._validate_common_inputs()
        if validated is None:
            return
        _source_dir, _workspace_dir, patch_size, downsample_factor = validated
        try:
            n_patches = int(self.train_count_var.get())
        except ValueError:
            messagebox.showerror("Invalid count", "Train patches must be an integer.")
            return
        if n_patches <= 0:
            messagebox.showerror("Invalid count", "Train patches must be > 0.")
            return

        source_map = self._source_map_for_scope()
        if not source_map:
            messagebox.showerror("No images", "No images found for current sampling scope.")
            return

        source_ids = sorted(source_map.keys())
        store = self._get_store()
        rng = random.Random()
        session_seed = rng.randint(1, 1_000_000_000)
        self._set_patch_progress(
            0,
            f"Generating 0/{n_patches} patches ({patch_size}px from {patch_size * downsample_factor}px source)",
        )

        created = 0
        skipped = 0
        overlap_skipped = 0
        attempts = 0
        max_attempts = max(n_patches * 200, 1000)
        accepted_rects_by_source: dict[str, list[tuple[int, int, int, int]]] = {}
        while created < n_patches and attempts < max_attempts:
            attempts += 1
            if attempts == 1 or attempts % 5 == 0:
                pct = (created / n_patches) * 100
                self._set_patch_progress(
                    pct,
                    f"Generating {created}/{n_patches} patches "
                    f"(attempt {attempts}, skipped {skipped}, overlaps {overlap_skipped})",
                )
            source_id = source_ids[attempts % len(source_ids)]
            src_path = rng.choice(source_map[source_id])
            img = tiff.imread(str(src_path))
            try:
                patch, x0, y0, source_w, source_h = _sample_patch_with_coords(
                    img,
                    patch_size=patch_size,
                    rng=rng,
                    downsample_factor=downsample_factor,
                )
            except ValueError:
                skipped += 1
                continue

            src_key = str(src_path.resolve())
            candidate_rect = (int(x0), int(y0), int(source_w), int(source_h))
            accepted_rects = accepted_rects_by_source.setdefault(src_key, [])
            if any(_rects_overlap(candidate_rect, existing) for existing in accepted_rects):
                overlap_skipped += 1
                continue

            source_hash = f"{src_path.stat().st_size}_{int(src_path.stat().st_mtime)}"
            with tempfile.TemporaryDirectory() as td:
                tmp_img = Path(td) / "tmp_img.tif"
                tmp_mask = Path(td) / "tmp_masks.tif"
                tiff.imwrite(str(tmp_img), patch)
                tiff.imwrite(str(tmp_mask), np.zeros((patch_size, patch_size), dtype=np.uint16))
                store.add_patch(
                    patch_img_path=tmp_img,
                    patch_mask_path=tmp_mask,
                    source_id=source_id,
                    source_image_path=src_path,
                    source_image_hash=source_hash,
                    x=x0,
                    y=y0,
                    w=source_w,
                    h=source_h,
                    seed=session_seed,
                    annotation_status="auto",
                    include=True,
                    split="unassigned",
                )
            created += 1
            accepted_rects.append(candidate_rect)
            self._set_patch_progress(
                (created / n_patches) * 100,
                f"Generating {created}/{n_patches} patches "
                f"(skipped {skipped}, overlaps {overlap_skipped})",
            )

        if created < n_patches:
            self._log(
                f"Generated {created}/{n_patches} train patches into patch pool "
                f"after {attempts} attempts (downsample={downsample_factor}x, "
                f"skipped small/incompatible: {skipped}, overlaps: {overlap_skipped})."
            )
            self._set_patch_progress(
                (created / n_patches) * 100,
                f"Generated {created}/{n_patches}; skipped {skipped}, overlaps {overlap_skipped}",
            )
        else:
            self._log(
                f"Generated {created} train patches into patch pool "
                f"(downsample={downsample_factor}x, skipped small/incompatible: {skipped}, "
                f"overlaps: {overlap_skipped})."
            )
            self._set_patch_progress(100, f"Generated {created}/{n_patches} patches")
        self.load_train_set()

    def _load_cpsam_model(self):
        try:
            from cellpose import models
        except Exception as exc:
            raise RuntimeError("Cellpose is not installed. Install with pip install cellpose") from exc

        gpu = bool(self.gpu_var.get())
        try:
            return models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
        except TypeError:
            return models.CellposeModel(gpu=gpu, model_type="cpsam")

    def autolabel_train_patches(self) -> None:
        items = self._load_patch_pairs()
        if not items:
            messagebox.showerror(
                "No train patches",
                "No patches found in manifest/pool.",
            )
            return

        def _job() -> None:
            self._log("Loading Cellpose-SAM model...")
            self._set_patch_progress_threadsafe(0, f"Loading Cellpose-SAM model for {len(items)} patches")
            try:
                model = self._load_cpsam_model()
            except Exception as exc:
                self._log(f"Auto-label failed: {exc}")
                self._set_patch_progress_threadsafe(0, "Auto-label failed")
                return

            self._set_patch_progress_threadsafe(0, f"Auto-labeling 0/{len(items)} patches")
            for idx, item in enumerate(items, start=1):
                self._set_patch_progress_threadsafe(
                    ((idx - 1) / len(items)) * 100,
                    f"Auto-labeling {idx}/{len(items)}: {item.patch_id}",
                )
                img = tiff.imread(str(item.image_path))
                dapi_idx = int(self.dapi_index_var.get().strip() or "0")
                gray = _extract_nuclei_channel(img, dapi_idx)
                try:
                    masks, _flows, _styles = model.eval(gray)
                except Exception as exc:
                    self._log(f"Failed on {item.image_path.name}: {exc}")
                    self._set_patch_progress_threadsafe(
                        (idx / len(items)) * 100,
                        f"Skipped {idx}/{len(items)} after error",
                    )
                    continue
                tiff.imwrite(str(item.mask_path), np.asarray(masks, dtype=np.uint16))
                self._log(f"[{idx}/{len(items)}] Wrote {item.mask_path.name}")
                self._set_patch_progress_threadsafe(
                    (idx / len(items)) * 100,
                    f"Auto-labeled {idx}/{len(items)} patches",
                )

            self._log("Auto-labeling complete.")
            self._set_patch_progress_threadsafe(100, f"Auto-labeled {len(items)}/{len(items)} patches")
            self.root.after(0, self.load_train_set)

        self._run_background(_job)

    def load_train_set(self) -> None:
        previous_patch_id = self.current_patch_id
        yview = self.patch_listbox.yview() if hasattr(self, "patch_listbox") else (0.0, 1.0)
        self.patch_items = self._load_patch_pairs()
        self.patch_listbox.delete(0, "end")
        for item in self.patch_items:
            self.patch_listbox.insert(
                "end",
                f"{item.patch_id} | {item.source_id} | {item.annotation_status} | {'in' if item.include else 'out'} | {item.split}",
            )

        if self.patch_items:
            target_index = 0
            if previous_patch_id:
                for i, item in enumerate(self.patch_items):
                    if item.patch_id == previous_patch_id:
                        target_index = i
                        break
            self.show_patch(target_index)
            try:
                self.patch_listbox.yview_moveto(yview[0])
            except Exception:
                pass
        else:
            self.current_index = -1
            self.current_patch_id = ""
            self.current_source_img = None
            self.current_img = None
            self.current_mask = None
            self.canvas.delete("all")
            self.patch_status_label.configure(text="No patch")

    def _on_patch_select(self, _event: tk.Event) -> None:
        selection = self.patch_listbox.curselection()
        if not selection:
            return
        # Use the most recently selected row and preserve multi-selection.
        self.show_patch(int(selection[-1]), sync_list_selection=False)

    def _selected_patch_ids(self) -> list[str]:
        sel = list(self.patch_listbox.curselection())
        patch_ids: list[str] = []
        for idx in sel:
            if 0 <= idx < len(self.patch_items):
                patch_ids.append(self.patch_items[idx].patch_id)
        if not patch_ids and self.current_patch_id:
            patch_ids = [self.current_patch_id]
        return patch_ids

    def _restore_patch_selection(self, patch_ids: list[str]) -> None:
        if not patch_ids or not hasattr(self, "patch_listbox"):
            return
        id_to_index = {item.patch_id: i for i, item in enumerate(self.patch_items)}
        indices = [id_to_index[pid] for pid in patch_ids if pid in id_to_index]
        if not indices:
            return
        self.patch_listbox.selection_clear(0, "end")
        for idx in indices:
            self.patch_listbox.selection_set(idx)
        self.patch_listbox.activate(indices[-1])
        self.show_patch(indices[-1], sync_list_selection=False)

    def _on_split_selected(self, _event: tk.Event | None = None) -> None:
        split = self.patch_split_var.get().strip()
        if split not in {"unassigned", "train", "val", "test"}:
            return
        patch_ids = self._selected_patch_ids()
        if not patch_ids:
            return
        store = self._get_store()
        for pid in patch_ids:
            store.update_patch_fields(pid, split=split)
        self._log(f"Set split='{split}' for {len(patch_ids)} selected patch(es).")
        self.load_train_set()
        self._restore_patch_selection(patch_ids)

    def show_patch(self, index: int, *, sync_list_selection: bool = True) -> None:
        if index < 0 or index >= len(self.patch_items):
            return
        item = self.patch_items[index]
        img = tiff.imread(str(item.image_path))
        rgb = _to_rgb_uint8(img)
        mask = tiff.imread(str(item.mask_path))
        if mask.shape[:2] != rgb.shape[:2]:
            messagebox.showerror("Shape mismatch", f"Mask shape mismatch for {item.image_path.name}")
            return

        self.current_index = index
        self.current_patch_id = item.patch_id
        self.current_source_img = np.asarray(img)
        self.current_img = rgb
        self.current_mask = np.asarray(mask, dtype=np.int32)
        self.patch_status_var.set(item.annotation_status)
        self.patch_include_var.set(item.include)
        self.patch_split_var.set(item.split)

        if sync_list_selection:
            self.patch_listbox.selection_clear(0, "end")
            self.patch_listbox.selection_set(index)
        self.patch_listbox.activate(index)
        self.patch_status_label.configure(text=f"Patch {index + 1}/{len(self.patch_items)}")
        self._render_current_patch()

    def _render_current_patch(self) -> None:
        self.canvas.delete("all")
        self.roi_draw_items.clear()
        if self.current_img is None:
            return

        if self.current_source_img is not None:
            try:
                low, high, gamma = self._get_contrast_params()
                channel_indices = self._get_channel_indices()
                rgb = _to_rgb_with_contrast(
                    self.current_source_img,
                    low_pct=low,
                    high_pct=high,
                    gamma=gamma,
                    show_dapi_gob_goa=(
                        bool(self.show_dapi_var.get()),
                        bool(self.show_gob_var.get()),
                        bool(self.show_goa_var.get()),
                    ),
                    channel_indices=channel_indices,
                )
                self.current_img = rgb
            except ValueError as exc:
                self._log(f"Display settings issue: {exc}")
                rgb = self.current_img
        else:
            rgb = self.current_img
        src_h, src_w = rgb.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return

        canvas_w = int(self.canvas.winfo_width())
        canvas_h = int(self.canvas.winfo_height())
        if canvas_w < 32 or canvas_h < 32:
            canvas_w = self.display_size
            canvas_h = self.display_size
        scale = min(canvas_w / src_w, canvas_h / src_h)
        disp_w = max(1, int(round(src_w * scale)))
        disp_h = max(1, int(round(src_h * scale)))
        x0 = (canvas_w - disp_w) // 2
        y0 = (canvas_h - disp_h) // 2
        self.current_display_x = x0
        self.current_display_y = y0
        self.current_display_w = disp_w
        self.current_display_h = disp_h

        if src_h != disp_h or src_w != disp_w:
            y_idx = np.linspace(0, src_h - 1, disp_h).astype(np.int32)
            x_idx = np.linspace(0, src_w - 1, disp_w).astype(np.int32)
            disp = rgb[np.ix_(y_idx, x_idx)]
        else:
            disp = rgb

        self.current_photo = tk.PhotoImage(data=_rgb_to_ppm_bytes(disp), format="PPM")
        self.canvas.create_image(x0, y0, anchor="nw", image=self.current_photo)

        if self.current_mask is None:
            return

        h, w = self.current_mask.shape
        sx = self.current_display_w / w
        sy = self.current_display_h / h

        for label_id in sorted(int(v) for v in np.unique(self.current_mask) if int(v) > 0):
            binary = self.current_mask == label_id
            contours = find_contours(binary.astype(np.uint8), level=0.5)
            for contour in contours:
                coords: list[float] = []
                for y, x in contour:
                    coords.extend([self.current_display_x + x * sx, self.current_display_y + y * sy])
                if len(coords) >= 4:
                    item = self.canvas.create_line(*coords, fill="#00ffff", width=2, smooth=True)
                    self.roi_draw_items.append(item)

            yy, xx = np.where(binary)
            if yy.size:
                cx = self.current_display_x + float(np.mean(xx)) * sx
                cy = self.current_display_y + float(np.mean(yy)) * sy
                text_item = self.canvas.create_text(
                    cx,
                    cy,
                    text="X",
                    fill="#ff3030",
                    font=("Arial", 12, "bold"),
                    tags=(f"delete_{label_id}", "delete"),
                )
                self.roi_draw_items.append(text_item)

    def prev_patch(self) -> None:
        if self.current_index > 0:
            self.show_patch(self.current_index - 1)

    def next_patch(self) -> None:
        if self.current_index >= 0 and self.current_index < len(self.patch_items) - 1:
            self.show_patch(self.current_index + 1)

    def save_current_mask(self) -> None:
        if self.current_index < 0 or self.current_mask is None or not self.current_patch_id:
            return
        item = self.patch_items[self.current_index]
        tiff.imwrite(str(item.mask_path), np.asarray(self.current_mask, dtype=np.uint16))
        if self.patch_status_var.get() == "auto":
            self.patch_status_var.set("edited")
            self._get_store().update_patch_fields(self.current_patch_id, annotation_status="edited")
        self._log(f"Saved {item.mask_path}")
        self.load_train_set()

    def delete_current_patch(self) -> None:
        if self.current_index < 0 or not self.current_patch_id:
            return
        patch_id = self.current_patch_id
        ok = self._get_store().remove_patch(patch_id)
        if ok:
            self._log(f"Deleted patch {patch_id}")
            self.load_train_set()
        else:
            self._log(f"Patch not found for delete: {patch_id}")

    def delete_selected_patches(self) -> None:
        sel = list(self.patch_listbox.curselection())
        if not sel:
            self.delete_current_patch()
            return
        patch_ids: list[str] = []
        for idx in sel:
            if 0 <= idx < len(self.patch_items):
                patch_ids.append(self.patch_items[idx].patch_id)
        if not patch_ids:
            return
        removed = 0
        for pid in patch_ids:
            if self._get_store().remove_patch(pid):
                removed += 1
        self._log(f"Deleted {removed}/{len(patch_ids)} selected patch(es).")
        self.load_train_set()

    def apply_patch_metadata(self) -> None:
        status = self.patch_status_var.get().strip()
        split = self.patch_split_var.get().strip()
        include = "true" if self.patch_include_var.get() else "false"
        patch_ids = self._selected_patch_ids()
        if not patch_ids:
            return

        for pid in patch_ids:
            self._get_store().update_patch_fields(
                pid,
                annotation_status=status,
                split=split,
                include=include,
            )
        self._log(f"Updated metadata for {len(patch_ids)} patch(es).")
        self.load_train_set()
        self._restore_patch_selection(patch_ids)

    def _delete_label(self, label_id: int) -> None:
        if self.current_mask is None or label_id <= 0:
            return
        self.current_mask[self.current_mask == label_id] = 0
        self._render_current_patch()

    def _get_contrast_params(self) -> tuple[float, float, float]:
        try:
            low = float(self.contrast_low_var.get().strip())
            high = float(self.contrast_high_var.get().strip())
            gamma = float(self.contrast_gamma_var.get().strip())
        except ValueError as exc:
            raise ValueError("Contrast values must be numeric.") from exc

        if not (0.0 <= low < 100.0 and 0.0 < high <= 100.0 and low < high):
            raise ValueError("Contrast percentiles must satisfy 0 <= low < high <= 100.")
        if gamma <= 0:
            raise ValueError("Gamma must be > 0.")
        return low, high, gamma

    def _get_channel_indices(self) -> tuple[int, int, int]:
        try:
            d = int(self.dapi_index_var.get().strip())
            g = int(self.gob_index_var.get().strip())
            a = int(self.goa_index_var.get().strip())
        except ValueError as exc:
            raise ValueError("Channel indices must be integers.") from exc
        if d < 0 or g < 0 or a < 0:
            raise ValueError("Channel indices must be >= 0.")
        return d, g, a

    def apply_contrast_settings(self) -> None:
        if self.current_img is None:
            return
        try:
            self._get_contrast_params()
            self._get_channel_indices()
        except ValueError as exc:
            messagebox.showerror("Invalid contrast settings", str(exc))
            return
        self._render_current_patch()

    def reset_contrast_settings(self) -> None:
        self.contrast_low_var.set("1")
        self.contrast_high_var.set("99")
        self.contrast_gamma_var.set("1.0")
        if self.current_img is not None:
            self._render_current_patch()

    def _next_label_id(self) -> int:
        if self.current_mask is None:
            return 1
        mx = int(np.max(self.current_mask))
        return mx + 1 if mx >= 0 else 1

    def _canvas_to_mask_xy(self, x: float, y: float) -> tuple[int, int] | None:
        if self.current_mask is None:
            return None
        h, w = self.current_mask.shape
        disp_w = max(1, int(self.current_display_w))
        disp_h = max(1, int(self.current_display_h))
        lx = x - self.current_display_x
        ly = y - self.current_display_y
        if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
            return None
        px = int(np.clip(lx / disp_w * w, 0, w - 1))
        py = int(np.clip(ly / disp_h * h, 0, h - 1))
        return px, py

    def _on_canvas_down(self, event: tk.Event) -> None:
        if self.current_mask is None:
            return

        overlap = self.canvas.find_overlapping(event.x - 3, event.y - 3, event.x + 3, event.y + 3)
        for item_id in overlap:
            tags = self.canvas.gettags(item_id)
            for tag in tags:
                if tag.startswith("delete_"):
                    label_id = int(tag.split("_", 1)[1])
                    self._delete_label(label_id)
                    return

        mode = self.mode_var.get()

        if mode == "point":
            xy = self._canvas_to_mask_xy(event.x, event.y)
            if xy is not None:
                self.add_mask_from_point(xy)

        elif mode == "freehand":
            self.freehand_points = [(event.x, event.y)]
            if self.freehand_line_item is not None:
                self.canvas.delete(self.freehand_line_item)
            self.freehand_line_item = self.canvas.create_line(
                event.x, event.y, event.x, event.y, fill="#ffd700", width=2
            )

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() != "freehand" or self.freehand_line_item is None:
            return
        self.freehand_points.append((event.x, event.y))
        coords: list[float] = []
        for x, y in self.freehand_points:
            coords.extend([x, y])
        self.canvas.coords(self.freehand_line_item, *coords)

    def _on_canvas_up(self, _event: tk.Event) -> None:
        if self.mode_var.get() != "freehand":
            return
        self._commit_freehand_roi()

    def _commit_freehand_roi(self) -> None:
        if self.current_mask is None or len(self.freehand_points) < 3:
            if self.freehand_line_item is not None:
                self.canvas.delete(self.freehand_line_item)
                self.freehand_line_item = None
            self.freehand_points = []
            return

        h, w = self.current_mask.shape
        disp_w = max(1, int(self.current_display_w))
        disp_h = max(1, int(self.current_display_h))
        xs = np.array(
            [np.clip((p[0] - self.current_display_x) / disp_w * w, 0, w - 1) for p in self.freehand_points],
            dtype=np.float32,
        )
        ys = np.array(
            [np.clip((p[1] - self.current_display_y) / disp_h * h, 0, h - 1) for p in self.freehand_points],
            dtype=np.float32,
        )
        rr, cc = polygon(ys, xs, shape=self.current_mask.shape)
        if rr.size == 0:
            self._log("Freehand ROI was empty after rasterization.")
        else:
            label_id = self._next_label_id()
            self.current_mask[rr, cc] = label_id
            self._log(f"Added freehand ROI label {label_id}")

        if self.freehand_line_item is not None:
            self.canvas.delete(self.freehand_line_item)
            self.freehand_line_item = None
        self.freehand_points = []
        self._render_current_patch()

    def _predict_sam2_mask(self, point_xy: tuple[int, int]) -> np.ndarray:
        try:
            from ultralytics import SAM
        except Exception as exc:
            raise RuntimeError("Ultralytics is not installed. Install with pip install ultralytics") from exc

        if self.current_img is None:
            raise RuntimeError("No image loaded")

        ckpt = self.ultra_sam2_ckpt_var.get().strip()
        if not ckpt:
            raise RuntimeError("Set a SAM2 checkpoint path/name")

        sam_model = SAM(ckpt)
        px, py = point_xy
        img = self.current_img

        attempts = [
            lambda: sam_model.predict(source=img, points=[[px, py]], labels=[1], verbose=False),
            lambda: sam_model.predict(
                source=img,
                points=np.array([[px, py]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
                verbose=False,
            ),
            lambda: sam_model(img, points=[[px, py]], labels=[1], verbose=False),
        ]

        last_exc: Exception | None = None
        results = None
        for fn in attempts:
            try:
                results = fn()
                break
            except Exception as exc:
                last_exc = exc

        if results is None:
            raise RuntimeError(f"SAM2 prediction failed: {last_exc}")

        if not results:
            raise RuntimeError("SAM2 returned no results")
        res0 = results[0]
        masks_obj = getattr(res0, "masks", None)
        if masks_obj is None:
            raise RuntimeError("SAM2 output had no masks")

        data = getattr(masks_obj, "data", None)
        if data is None:
            raise RuntimeError("SAM2 masks missing data")

        try:
            masks_np = data.cpu().numpy()
        except Exception:
            masks_np = np.asarray(data)

        if masks_np.ndim == 2:
            mask = masks_np
        elif masks_np.ndim == 3 and masks_np.shape[0] >= 1:
            mask = masks_np[0]
        else:
            raise RuntimeError(f"Unexpected SAM2 mask shape: {masks_np.shape}")

        return np.asarray(mask > 0.5, dtype=bool)

    def add_mask_from_point(self, point_xy: tuple[int, int]) -> None:
        if self.current_mask is None:
            return

        def _job() -> None:
            self._log(f"Running SAM2 point prompt at {point_xy}...")
            try:
                mask = self._predict_sam2_mask(point_xy)
            except Exception as exc:
                self._log(f"SAM2 point mode failed: {exc}")
                return

            if mask.shape != self.current_mask.shape:
                self._log(
                    f"SAM2 mask shape mismatch: got {mask.shape}, expected {self.current_mask.shape}"
                )
                return

            label_id = self._next_label_id()
            self.current_mask[mask] = label_id
            self._log(f"Added SAM2 mask as label {label_id}")
            self.root.after(0, self._render_current_patch)

        self._run_background(_job)

    def build_snapshot(self) -> None:
        if not self.workspace_dir_var.get().strip():
            messagebox.showerror("Missing workspace", "Set Workspace Folder first.")
            return
        name = self.snapshot_name_var.get().strip() or "snapshot"
        store = self._get_store()
        selected_ids = self._selected_source_ids()
        source_ids = selected_ids if selected_ids else [s.source_id for s in store.load_sources() if s.active]
        if not source_ids:
            messagebox.showerror("No sources", "Add at least one source or select sources.")
            return
        try:
            nuclei_idx = int(self.dapi_index_var.get().strip())
            if nuclei_idx < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid DAPI index", "DAPI index must be a non-negative integer.")
            return
        try:
            snap_id = store.build_snapshot(
                name=name,
                source_ids=source_ids,
                approved_only=False,
                include_only=True,
                split_ratios=(0.8, 0.1, 0.1),
                stratify_by_source=True,
                seed=42,
                nuclei_channel_index=nuclei_idx,
            )
        except Exception as exc:
            messagebox.showerror("Snapshot failed", str(exc))
            return
        self._refresh_snapshot_choices()
        self._set_snapshot_choice(snap_id)
        self._log(f"Built snapshot {snap_id}")

    def delete_snapshot(self) -> None:
        snapshot_id = self.snapshot_choice_var.get().strip()
        if not snapshot_id:
            display = self.snapshot_display_var.get().strip()
            snapshot_id = self.snapshot_display_to_id.get(display, "")
        if not snapshot_id:
            messagebox.showerror("No snapshot", "Select a snapshot to delete.")
            return
        display_name = self.snapshot_id_to_display.get(snapshot_id, snapshot_id)
        if not messagebox.askyesno(
            "Delete snapshot",
            f"Delete snapshot '{display_name}'?\nThis removes its train/val/test exports.",
        ):
            return
        try:
            deleted = self._get_store().delete_snapshot(snapshot_id)
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        if not deleted:
            messagebox.showerror("Delete failed", f"Snapshot not found: {snapshot_id}")
            return
        self._refresh_snapshot_choices()
        self._log(f"Deleted snapshot {snapshot_id}")

    def populate_train_command(self) -> None:
        try:
            cfg = self._build_train_api_config()
        except Exception as exc:
            messagebox.showerror("Training setup failed", str(exc))
            return
        self.train_cmd_var.set(self._format_train_api_summary(cfg))
        if cfg["n_test"] > 0:
            self._log(f"API training will use validation data ({cfg['n_test']} labeled pair(s)).")
        else:
            self._log("No labeled validation pairs found; API training will run without test_data.")
        self._log("Populated Cellpose API training call.")

    def run_training_command(self) -> None:
        try:
            cfg = self._build_train_api_config()
        except Exception as exc:
            messagebox.showerror("Training setup failed", str(exc))
            return
        api_summary = self._format_train_api_summary(cfg)
        self.train_cmd_var.set(api_summary)

        params = {
            "learning_rate": self.learning_rate_var.get().strip(),
            "weight_decay": self.weight_decay_var.get().strip(),
            "n_epochs": self.n_epochs_var.get().strip(),
            "train_batch_size": self.train_batch_size_var.get().strip(),
            "min_train_masks": self.min_train_masks_var.get().strip(),
            "verbose": "true" if self.train_verbose_var.get() else "false",
            "best_val_checkpoint": "true" if self.train_best_val_var.get() else "false",
            "model_name_out": self.model_name_out_var.get().strip(),
            "api": "cellpose.train.train_seg",
        }
        run_id = self._get_store().create_run(
            snapshot_id=cfg["snapshot_id"] or "none",
            train_command=api_summary,
            params=params,
        )
        self.run_id_active = run_id
        run_log_path = self.workspace_dir / "runs" / run_id / "train.log"

        def _job() -> None:
            self.training_cancel_requested = False
            self.training_active = True
            self._log(f"Running Cellpose API training:\n{api_summary}")
            try:
                with run_log_path.open("w", encoding="utf-8") as log_fh:
                    model_output_path = self._run_cellpose_training_api(cfg, log_fh)
                if self.training_cancel_requested:
                    self._log("Cancel was requested; Cellpose API call has returned.")
                    self._get_store().finalize_run(run_id, status="canceled")
                else:
                    model_output_path = self._postprocess_training_output(
                        run_log_path, api_model_path=model_output_path
                    )
                    self._log("Training completed.")
                    self._get_store().finalize_run(
                        run_id, status="success", model_output_path=model_output_path
                    )
            except Exception as exc:
                self._log(f"Cellpose API training failed: {exc}")
                self._get_store().finalize_run(run_id, status="failed")
            finally:
                self.training_active = False
                self.training_cancel_requested = False
                self.run_id_active = None
                self.root.after(0, self._refresh_runs_list)

        self._run_background(_job)

    def _build_train_api_config(self) -> dict:
        snapshot_id = self.snapshot_choice_var.get().strip()
        if not snapshot_id:
            raise RuntimeError(
                "Build and select a snapshot first. Snapshots export nuclei-channel-only images for training."
            )
        snap_dir = self.workspace_dir / "snapshots" / snapshot_id
        train_root = snap_dir / "train"
        val_root = snap_dir / "val"

        n_train = self._count_labeled_pairs(train_root)
        n_test = self._count_labeled_pairs(val_root)
        if n_train == 0:
            raise RuntimeError(
                f"No labeled train pairs found in {train_root}.\n"
                "Need files like *_img.tif and matching *_masks.tif."
            )

        return {
            "snapshot_id": snapshot_id,
            "train_root": train_root,
            "val_root": val_root,
            "n_train": n_train,
            "n_test": n_test,
            "gpu": bool(self.gpu_var.get()),
            "learning_rate": float(self.learning_rate_var.get().strip()),
            "weight_decay": float(self.weight_decay_var.get().strip()),
            "n_epochs": int(self.n_epochs_var.get().strip()),
            "train_batch_size": int(self.train_batch_size_var.get().strip()),
            "min_train_masks": int(self.min_train_masks_var.get().strip()),
            "model_name_out": self.model_name_out_var.get().strip(),
            "save_each": bool(self.train_best_val_var.get()),
            "verbose": bool(self.train_verbose_var.get()),
        }

    @staticmethod
    def _format_train_api_summary(cfg: dict) -> str:
        summary = (
            "cellpose.train.train_seg("
            f"train_data={cfg['n_train']} image(s), "
            f"test_data={cfg['n_test']} image(s), "
            f"learning_rate={cfg['learning_rate']}, "
            f"weight_decay={cfg['weight_decay']}, "
            f"n_epochs={cfg['n_epochs']}, "
            f"batch_size={cfg['train_batch_size']}, "
            f"min_train_masks={cfg['min_train_masks']}, "
            f"model_name={cfg['model_name_out']}, "
            f"gpu={cfg['gpu']}"
        )
        if cfg["save_each"]:
            summary += f", save_each=True, save_every={BEST_VAL_SAVE_EVERY}"
        summary += ")"
        return summary

    @staticmethod
    def _load_labeled_pairs(folder: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for img_path in sorted(folder.glob("*_img.tif")):
            mask_path = img_path.with_name(img_path.name.replace("_img.tif", "_masks.tif"))
            if not mask_path.exists():
                continue
            images.append(np.asarray(tiff.imread(str(img_path))))
            labels.append(np.asarray(tiff.imread(str(mask_path)), dtype=np.int32))
        return images, labels

    def _run_cellpose_training_api(self, cfg: dict, log_fh) -> str:
        try:
            from cellpose import models, train
        except Exception as exc:
            raise RuntimeError("Cellpose is not installed. Install with pip install cellpose") from exc

        train_data, train_labels = self._load_labeled_pairs(cfg["train_root"])
        test_data: list[np.ndarray] | None = None
        test_labels: list[np.ndarray] | None = None
        if cfg["n_test"] > 0:
            test_data, test_labels = self._load_labeled_pairs(cfg["val_root"])

        model = models.CellposeModel(gpu=cfg["gpu"])
        train_seg = train.train_seg
        accepted = set(inspect.signature(train_seg).parameters)
        kwargs = {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
            "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
            "n_epochs": cfg["n_epochs"],
            "min_train_masks": cfg["min_train_masks"],
            "save_path": str(cfg["train_root"].resolve()),
            "save_each": cfg["save_each"],
            "save_every": BEST_VAL_SAVE_EVERY,
        }
        if "batch_size" in accepted:
            kwargs["batch_size"] = cfg["train_batch_size"]
        elif "train_batch_size" in accepted:
            kwargs["train_batch_size"] = cfg["train_batch_size"]
        if "model_name" in accepted:
            kwargs["model_name"] = cfg["model_name_out"]
        elif "model_name_out" in accepted:
            kwargs["model_name_out"] = cfg["model_name_out"]
        if not test_data:
            kwargs.pop("test_data", None)
            kwargs.pop("test_labels", None)
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        skipped = sorted(set(kwargs) - set(filtered))

        writer = _TrainingLogWriter(self._log, log_fh)
        writer.write(f"Loaded {len(train_data)} train pair(s)\n")
        if test_data is not None:
            writer.write(f"Loaded {len(test_data)} validation pair(s)\n")
        if skipped:
            writer.write(f"Skipped unsupported Cellpose train_seg parameter(s): {', '.join(skipped)}\n")
        handler = logging.StreamHandler(writer)
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        if cfg["verbose"]:
            root_logger.setLevel(logging.INFO)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = train_seg(model.net, **filtered)
        finally:
            writer.flush()
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
        return self._extract_model_path_from_train_result(
            result,
            train_root=cfg["train_root"],
            model_name=cfg["model_name_out"],
        )

    @staticmethod
    def _extract_model_path_from_train_result(result, *, train_root: Path, model_name: str) -> str:
        candidates: list[Path] = []
        if isinstance(result, (str, Path)):
            candidates.append(Path(result))
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, (str, Path)):
                    candidates.append(Path(item))

        model_dir = train_root / "models"
        candidates.append(model_dir / model_name)
        if model_dir.exists():
            candidates.extend(sorted(model_dir.glob(f"{model_name}*"), key=lambda p: p.stat().st_mtime, reverse=True))

        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if resolved.exists():
                return str(resolved)
        return ""

    def _postprocess_training_output(self, run_log_path: Path, api_model_path: str = "") -> str:
        if api_model_path:
            model_path = Path(api_model_path)
            if model_path.exists():
                resolved = str(model_path.resolve())
                self.trained_model_path_var.set(resolved)
                return resolved

        if not run_log_path.exists():
            return ""
        try:
            lines = run_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""

        base_model_path = ""
        re_base = re.compile(r"saving model to (.+)$")
        re_loss = re.compile(r"^\s*(\d+),\s*train_loss=.*?test_loss=([0-9]*\.?[0-9]+)")
        best_epoch: int | None = None
        best_test_loss: float | None = None
        scored_epochs: list[tuple[float, int]] = []
        for line in lines:
            m_base = re_base.search(line)
            if m_base:
                base_model_path = m_base.group(1).strip()
            m_loss = re_loss.search(line)
            if not m_loss:
                continue
            epoch = int(m_loss.group(1))
            test_loss = float(m_loss.group(2))
            scored_epochs.append((test_loss, epoch))
            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch

        candidate = ""
        if self.train_best_val_var.get() and base_model_path and scored_epochs:
            for loss_val, epoch_val in sorted(scored_epochs, key=lambda x: x[0]):
                epoch_path = Path(f"{base_model_path}_epoch_{epoch_val:04d}")
                if epoch_path.exists():
                    candidate = str(epoch_path.resolve())
                    best_epoch = epoch_val
                    best_test_loss = loss_val
                    self._log(
                        f"Selected best val checkpoint: epoch {best_epoch} (test_loss={best_test_loss:.4f})"
                    )
                    break
            if not candidate and best_epoch is not None:
                self._log(
                    f"Best epoch {best_epoch} reported, but checkpoint file not found; using final model."
                )

        if not candidate and base_model_path:
            final_path = Path(base_model_path)
            if final_path.exists():
                candidate = str(final_path.resolve())

        if candidate:
            self.trained_model_path_var.set(candidate)
        return candidate

    def cancel_training(self) -> None:
        if not self.training_active:
            self._log("No active training run to cancel.")
            return
        self.training_cancel_requested = True
        self._log(
            "Cancel requested. Cellpose API training runs in-process, so it will stop only when the API call returns."
        )

    def _load_trained_model(self):
        try:
            from cellpose import models
        except Exception as exc:
            raise RuntimeError("Cellpose is not installed. Install with pip install cellpose") from exc

        model_path = self.trained_model_path_var.get().strip()
        if not model_path:
            raise RuntimeError("Set Trained model path.")
        path_obj = Path(model_path).expanduser().resolve()
        if not path_obj.exists():
            raise RuntimeError(f"Model path does not exist: {path_obj}")

        gpu = bool(self.gpu_var.get())
        return models.CellposeModel(gpu=gpu, pretrained_model=str(path_obj))

    def run_eval_inference(self) -> None:
        try:
            eval_dir = self._eval_input_dir()
        except RuntimeError as exc:
            messagebox.showerror("No snapshot test set", str(exc))
            return
        eval_imgs = sorted(eval_dir.glob("*_img.tif"))
        if not eval_imgs:
            messagebox.showerror(
                "No test patches",
                f"No *_img.tif in {eval_dir}. Mark some patches as split='test' and rebuild the snapshot.",
            )
            return

        _ensure_dir(self.eval_results_dir)
        for old in self.eval_results_dir.glob("eval_*_img_pred_masks.tif"):
            _safe_unlink(old)
        first_preview: dict[str, np.ndarray] = {}

        def _job() -> None:
            self._log("Loading trained model for evaluation...")
            try:
                model = self._load_trained_model()
            except Exception as exc:
                self._log(f"Eval failed: {exc}")
                return

            for i, img_path in enumerate(eval_imgs, start=1):
                img = tiff.imread(str(img_path))
                dapi_idx = int(self.dapi_index_var.get().strip() or "0")
                gray = _extract_nuclei_channel(img, dapi_idx)
                try:
                    masks, _flows, _styles = model.eval(gray)
                except Exception as exc:
                    self._log(f"Eval failed on {img_path.name}: {exc}")
                    continue

                out_mask = self.eval_results_dir / f"{img_path.stem}_pred_masks.tif"
                tiff.imwrite(str(out_mask), np.asarray(masks, dtype=np.uint16))
                self._log(f"[{i}/{len(eval_imgs)}] Saved {out_mask.name}")
                if not first_preview:
                    mask_arr = np.asarray(masks, dtype=np.int32)
                    meta_path = self._eval_meta_path_for_image(img_path)
                    display_img = self._display_image_for_eval(
                        eval_img_path=img_path,
                        eval_img=np.asarray(img),
                        mask=mask_arr,
                        meta_path=meta_path if meta_path.exists() else None,
                    )
                    first_preview["img"] = display_img
                    first_preview["mask"] = mask_arr

            self._log(f"Eval finished. Results in {self.eval_results_dir}")
            if first_preview:
                self.root.after(
                    0, lambda: self._preview_eval_result(first_preview["img"], first_preview["mask"])
                )
            self.root.after(0, self.load_eval_results)

        self._run_background(_job)

    def _preview_eval_result(self, img: np.ndarray, mask: np.ndarray) -> None:
        self.current_index = -1
        self.current_patch_id = ""
        self.current_eval_img_path = None
        self.current_eval_pred_mask_path = None
        self.current_source_img = np.asarray(img)
        self.current_img = _to_rgb_uint8(np.asarray(img))
        self.current_mask = np.asarray(mask, dtype=np.int32)
        self.patch_listbox.selection_clear(0, "end")
        self.patch_status_label.configure(text="Eval preview")
        self._render_current_patch()

    def _eval_input_dir(self) -> Path:
        snapshot_id = self.snapshot_choice_var.get().strip()
        if not snapshot_id:
            raise RuntimeError("Select a dataset snapshot first.")
        return self.workspace_dir / "snapshots" / snapshot_id / "test"

    def _eval_meta_path_for_image(self, img_path: Path) -> Path:
        return img_path.with_name(img_path.name.replace("_img.tif", "_meta.json"))

    def _collect_eval_result_items(self) -> list[tuple[Path, Path, Path | None]]:
        eval_dir = self._eval_input_dir()
        eval_imgs = sorted(eval_dir.glob("*_img.tif"))
        items: list[tuple[Path, Path, Path | None]] = []
        for img_path in eval_imgs:
            pred_mask_path = self.eval_results_dir / f"{img_path.stem}_pred_masks.tif"
            if pred_mask_path.exists():
                meta_path = self._eval_meta_path_for_image(img_path)
                items.append((img_path, pred_mask_path, meta_path if meta_path.exists() else None))
        return items

    @staticmethod
    def _is_multichannel_image(img: np.ndarray) -> bool:
        arr = np.asarray(img)
        if arr.ndim != 3:
            return False
        if arr.shape[-1] in (3, 4):
            return True
        if arr.shape[0] in (3, 4):
            return True
        return False

    def _show_eval_result_at(self, index: int) -> None:
        if index < 0 or index >= len(self.eval_preview_items):
            return
        img_path, pred_mask_path, meta_path = self.eval_preview_items[index]
        cache_key = str(img_path.resolve())
        if cache_key in self.eval_display_cache:
            display_img, mask_i32 = self.eval_display_cache[cache_key]
        else:
            try:
                img = tiff.imread(str(img_path))
                mask = tiff.imread(str(pred_mask_path))
            except Exception as exc:
                self._log(f"Failed to load eval preview: {exc}")
                return
            mask_i32 = np.asarray(mask, dtype=np.int32)
            if self._is_multichannel_image(img):
                display_img = np.asarray(img)
            else:
                display_img = self._display_image_for_eval(
                    eval_img_path=img_path,
                    eval_img=np.asarray(img),
                    mask=mask_i32,
                    meta_path=meta_path,
                )
            self.eval_display_cache[cache_key] = (np.asarray(display_img), mask_i32)
        self.eval_preview_index = index
        self.eval_preview_status_var.set(f"Eval results: {index + 1}/{len(self.eval_preview_items)}")
        self.current_index = -1
        self.current_patch_id = ""
        self.current_eval_img_path = img_path
        self.current_eval_pred_mask_path = pred_mask_path
        self.current_source_img = np.asarray(display_img)
        self.current_img = _to_rgb_uint8(np.asarray(display_img))
        self.current_mask = np.asarray(mask_i32, dtype=np.int32)
        self.patch_listbox.selection_clear(0, "end")
        if hasattr(self, "eval_listbox"):
            self.eval_listbox.selection_clear(0, "end")
            self.eval_listbox.selection_set(index)
            self.eval_listbox.activate(index)
        self.patch_status_label.configure(text=f"Eval preview {index + 1}/{len(self.eval_preview_items)}")
        self._render_current_patch()

    def load_eval_results(self) -> None:
        self.eval_preview_items = self._collect_eval_result_items()
        self.eval_display_cache.clear()
        if hasattr(self, "eval_listbox"):
            self.eval_listbox.delete(0, "end")
            for img_path, _pred_path, _meta in self.eval_preview_items:
                self.eval_listbox.insert("end", img_path.name)
        if not self.eval_preview_items:
            self.eval_preview_index = -1
            self.eval_preview_status_var.set("Eval results: 0")
            self._log("No eval result pairs found. Run eval inference first.")
            return
        self._show_eval_result_at(0)

    def _on_eval_select(self, _event: tk.Event | None = None) -> None:
        if not hasattr(self, "eval_listbox"):
            return
        sel = self.eval_listbox.curselection()
        if not sel:
            return
        self._show_eval_result_at(int(sel[-1]))

    def prev_eval_result(self) -> None:
        if not self.eval_preview_items:
            self.load_eval_results()
            return
        new_index = (self.eval_preview_index - 1) % len(self.eval_preview_items)
        self._show_eval_result_at(new_index)

    def next_eval_result(self) -> None:
        if not self.eval_preview_items:
            self.load_eval_results()
            return
        new_index = (self.eval_preview_index + 1) % len(self.eval_preview_items)
        self._show_eval_result_at(new_index)

    def _find_manifest_row_for_eval_img(self, eval_img_path: Path) -> dict[str, str] | None:
        name = eval_img_path.name
        if not name.endswith("_img.tif") or not name.startswith("patch_"):
            return None
        patch_id = name.replace("_img.tif", "")
        rows = self._get_store().read_manifest()
        for row in rows:
            if row.get("patch_id", "") == patch_id:
                return row
        return None

    def _eval_provenance(self, eval_img_path: Path, meta_path: Path | None) -> dict[str, str] | None:
        if meta_path is not None and meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                return {k: str(v) for k, v in payload.items()}
            except Exception:
                pass

        row = self._find_manifest_row_for_eval_img(eval_img_path)
        if row is None:
            return None
        return {
            "source_id": str(row.get("source_id", "")),
            "source_image_path": str(row.get("source_image_path", "")),
            "source_image_hash": str(row.get("source_image_hash", "")),
            "x": str(row.get("x", "0")),
            "y": str(row.get("y", "0")),
            "w": str(row.get("w", "0")),
            "h": str(row.get("h", "0")),
            "seed": str(row.get("seed", "0")),
        }

    @staticmethod
    def _crop_patch_from_source(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim < 2:
            raise ValueError(f"Unsupported source shape: {arr.shape}")
        if arr.ndim == 2:
            h_img, w_img = arr.shape
            x0 = max(0, min(int(x), w_img - 1))
            y0 = max(0, min(int(y), h_img - 1))
            ww = max(1, min(int(w), w_img - x0))
            hh = max(1, min(int(h), h_img - y0))
            return np.asarray(arr[y0 : y0 + hh, x0 : x0 + ww])
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            h_img, w_img = arr.shape[1], arr.shape[2]
            x0 = max(0, min(int(x), w_img - 1))
            y0 = max(0, min(int(y), h_img - 1))
            ww = max(1, min(int(w), w_img - x0))
            hh = max(1, min(int(h), h_img - y0))
            return np.asarray(arr[:, y0 : y0 + hh, x0 : x0 + ww])
        h_img, w_img = arr.shape[0], arr.shape[1]
        x0 = max(0, min(int(x), w_img - 1))
        y0 = max(0, min(int(y), h_img - 1))
        ww = max(1, min(int(w), w_img - x0))
        hh = max(1, min(int(h), h_img - y0))
        return np.asarray(arr[y0 : y0 + hh, x0 : x0 + ww, ...])

    def _display_image_for_eval(
        self, eval_img_path: Path, eval_img: np.ndarray, mask: np.ndarray, meta_path: Path | None
    ) -> np.ndarray:
        provenance = self._eval_provenance(eval_img_path, meta_path)
        if provenance is None:
            return np.asarray(eval_img)
        source_path_val = str(provenance.get("source_image_path", "")).strip()
        if not source_path_val:
            return np.asarray(eval_img)
        source_path = Path(source_path_val).expanduser().resolve()
        if not source_path.exists():
            return np.asarray(eval_img)
        try:
            src = tiff.imread(str(source_path))
            x = int(str(provenance.get("x", "0")).strip() or "0")
            y = int(str(provenance.get("y", "0")).strip() or "0")
            w = int(str(provenance.get("w", "0")).strip() or "0")
            h = int(str(provenance.get("h", "0")).strip() or "0")
            if w <= 0 or h <= 0:
                return np.asarray(eval_img)
            patch = self._crop_patch_from_source(src, x=x, y=y, w=w, h=h)
        except Exception:
            return np.asarray(eval_img)

        try:
            if patch.ndim == 2:
                patch_hw = patch.shape
            elif patch.ndim == 3 and patch.shape[0] in (3, 4) and patch.shape[-1] not in (3, 4):
                patch_hw = (patch.shape[1], patch.shape[2])
            else:
                patch_hw = (patch.shape[0], patch.shape[1])
            if tuple(patch_hw) != tuple(mask.shape[:2]):
                patch = _resize_patch_to_size(np.asarray(patch), int(mask.shape[0]))
        except Exception:
            return np.asarray(eval_img)
        return np.asarray(patch)

    def add_current_eval_to_train_set(self) -> None:
        if self.eval_preview_index < 0 or self.eval_preview_index >= len(self.eval_preview_items):
            messagebox.showerror("No eval selected", "Load eval results and select an eval patch first.")
            return
        eval_img_path, pred_mask_path, meta_path = self.eval_preview_items[self.eval_preview_index]
        if not eval_img_path.exists() or not pred_mask_path.exists():
            messagebox.showerror("Missing files", "Selected eval image or predicted mask is missing.")
            return

        provenance = self._eval_provenance(eval_img_path, meta_path)
        if provenance is None:
            messagebox.showerror(
                "Missing provenance",
                "Cannot map this eval patch back to its source image. Regenerate eval patches and rerun inference.",
            )
            return

        try:
            patch_img = tiff.imread(str(eval_img_path))
            patch_mask = tiff.imread(str(pred_mask_path)).astype(np.uint16)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not read eval files: {exc}")
            return

        source_path_val = provenance.get("source_image_path", "").strip()
        if not source_path_val:
            messagebox.showerror("Missing source", "Eval metadata is missing source image path.")
            return
        source_image_path = Path(source_path_val).expanduser().resolve()

        source_id = provenance.get("source_id", "").strip()
        if not source_id:
            if source_image_path.exists():
                source_id = self._get_store().upsert_source(source_image_path).source_id
            else:
                messagebox.showerror("Missing source", f"Source image path does not exist: {source_image_path}")
                return

        source_hash = provenance.get("source_image_hash", "").strip()
        if not source_hash and source_image_path.exists():
            source_hash = f"{source_image_path.stat().st_size}_{int(source_image_path.stat().st_mtime)}"
        if not source_hash:
            source_hash = "unknown"

        def _to_int(val: str, default: int) -> int:
            try:
                return int(str(val).strip())
            except Exception:
                return default

        h, w = patch_mask.shape[:2]
        x = _to_int(provenance.get("x", "0"), 0)
        y = _to_int(provenance.get("y", "0"), 0)
        ww = _to_int(provenance.get("w", str(w)), w)
        hh = _to_int(provenance.get("h", str(h)), h)
        seed = _to_int(provenance.get("seed", "0"), 0)

        with tempfile.TemporaryDirectory() as td:
            tmp_img = Path(td) / "add_eval_img.tif"
            tmp_mask = Path(td) / "add_eval_masks.tif"
            tiff.imwrite(str(tmp_img), patch_img)
            tiff.imwrite(str(tmp_mask), patch_mask)
            new_patch_id = self._get_store().add_patch(
                patch_img_path=tmp_img,
                patch_mask_path=tmp_mask,
                source_id=source_id,
                source_image_path=source_image_path,
                source_image_hash=source_hash,
                x=x,
                y=y,
                w=ww,
                h=hh,
                seed=seed,
                annotation_status="edited",
                include=True,
                split="unassigned",
            )

        self._log(f"Added eval prediction to training set as {new_patch_id}.")
        self.load_train_set()

    def _run_background(self, job) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A background task is still running.")
            return

        self.worker_thread = threading.Thread(target=job, daemon=True)
        self.worker_thread.start()

    def run(self) -> None:
        self.root.mainloop()


def launch_cpsam_finetune_gui() -> None:
    app = CpsamFineTuneApp()
    app.run()


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Tkinter interface for Cellpose-SAM fine-tuning")


def main() -> None:  # pragma: no cover
    _build_parser().parse_args()
    launch_cpsam_finetune_gui()


if __name__ == "__main__":  # pragma: no cover
    main()
