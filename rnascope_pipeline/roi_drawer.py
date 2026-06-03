"""Interactive Tkinter tool to draw freehand ImageJ/Fiji ROI files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox

import numpy as np
import roifile
import tifffile as tiff


LUT_NAMES = ("gray", "red", "green", "blue", "orange", "cyan", "magenta", "yellow", "white")


@dataclass
class _DisplayContext:
    source_w: int
    source_h: int
    display_w: int
    display_h: int

    @property
    def x_scale(self) -> float:
        return self.source_w / self.display_w

    @property
    def y_scale(self) -> float:
        return self.source_h / self.display_h


@dataclass
class _ChannelUi:
    enabled: tk.BooleanVar
    lut: tk.StringVar
    low_pct: tk.DoubleVar
    high_pct: tk.DoubleVar


def _infer_hwc_channels(image: np.ndarray) -> np.ndarray:
    """Convert image to HWC float32 channels for visualization."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[..., None].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected 2-D or 3-D TIFF.")

    if arr.shape[-1] in (2, 3, 4):
        out = arr[..., : min(arr.shape[-1], 4)]
    elif arr.shape[0] in (2, 3, 4):
        out = np.transpose(arr[: min(arr.shape[0], 4), ...], (1, 2, 0))
    else:
        # Fallback for unusual stacks: max-project to a single display channel.
        out = np.max(arr, axis=0)[..., None]
    return out.astype(np.float32)


def _resize_nearest_hwc(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize HWC image with nearest-neighbor sampling using NumPy only."""
    h, w = image.shape[:2]
    if out_h == h and out_w == w:
        return image
    y_idx = np.linspace(0, h - 1, out_h).astype(np.int32)
    x_idx = np.linspace(0, w - 1, out_w).astype(np.int32)
    return image[np.ix_(y_idx, x_idx)]


def _stretch_channel_to_uint8(channel: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Apply percentile contrast to one channel and return uint8."""
    low_pct = float(np.clip(low_pct, 0.0, 100.0))
    high_pct = float(np.clip(high_pct, 0.0, 100.0))
    if high_pct <= low_pct:
        return np.zeros(channel.shape, dtype=np.uint8)
    lo, hi = np.percentile(channel, (low_pct, high_pct))
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    return np.clip((channel - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _apply_lut(gray: np.ndarray, lut_name: str) -> np.ndarray:
    """Map grayscale uint8 image to RGB using the selected LUT."""
    g = gray.astype(np.uint8)
    z = np.zeros_like(g, dtype=np.uint8)
    if lut_name == "gray":
        return np.stack([g, g, g], axis=-1)
    if lut_name == "red":
        return np.stack([g, z, z], axis=-1)
    if lut_name == "green":
        return np.stack([z, g, z], axis=-1)
    if lut_name == "blue":
        return np.stack([z, z, g], axis=-1)
    if lut_name == "orange":
        green = np.clip((g.astype(np.float32) * (165.0 / 255.0)), 0, 255).astype(np.uint8)
        return np.stack([g, green, z], axis=-1)
    if lut_name == "cyan":
        return np.stack([z, g, g], axis=-1)
    if lut_name == "magenta":
        return np.stack([g, z, g], axis=-1)
    if lut_name == "yellow":
        return np.stack([g, g, z], axis=-1)
    if lut_name == "white":
        return np.stack([g, g, g], axis=-1)
    return np.stack([g, g, g], axis=-1)


def _write_ppm(rgb: np.ndarray, path: Path) -> None:
    """Write RGB uint8 array as binary PPM image."""
    if rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 image for PPM output.")
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError("Expected RGB image.")
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes(order="C"))


def _clean_polyline(points: list[tuple[float, float]]) -> np.ndarray:
    """Drop duplicate sequential points and return Nx2 float array."""
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float64)
    cleaned = [points[0]]
    for pt in points[1:]:
        if pt != cleaned[-1]:
            cleaned.append(pt)
    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(cleaned, dtype=np.float64)


def _save_freehand_roi(
    points_xy: np.ndarray,
    out_path: Path,
    *,
    roi_name: str,
    image_width: int,
    image_height: int,
) -> None:
    """Save points as a Fiji/ImageJ freehand ROI file."""
    if points_xy.shape[0] < 3:
        raise ValueError("Need at least 3 points to save a freehand ROI.")

    pts = np.round(points_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, image_width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, image_height - 1)

    dedup = [pts[0]]
    for p in pts[1:]:
        if p[0] != dedup[-1][0] or p[1] != dedup[-1][1]:
            dedup.append(p)
    if len(dedup) < 3:
        raise ValueError("ROI collapsed after rounding; draw a larger loop.")

    roi = roifile.ImagejRoi.frompoints(np.asarray(dedup), name=roi_name)
    roi.version = 228
    roi.options = roifile.ROI_OPTIONS(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    roi.tofile(str(out_path))


def _default_channel_names(n_channels: int) -> list[str]:
    if n_channels == 1:
        return ["Signal"]
    if n_channels == 2:
        return ["Channel1", "Channel2"]
    if n_channels >= 3:
        names = ["DAPI", "GoB", "GoA"]
        while len(names) < n_channels:
            names.append(f"Channel{len(names) + 1}")
        return names[:n_channels]
    return []


def _default_lut_for_name(name: str) -> str:
    lname = name.lower()
    if "dapi" in lname:
        return "gray"
    if "gob" in lname:
        return "orange"
    if "goa" in lname:
        return "red"
    return "white"


class FreehandRoiApp:
    """Simple GUI for drawing and exporting one freehand ROI."""

    def __init__(self, image_path: Path, output_roi: Path, roi_name: str | None = None) -> None:
        self.image_path = image_path
        self.output_roi = output_roi
        self.roi_name = roi_name or output_roi.stem or "roi"

        image = tiff.imread(str(image_path))
        self.source_hwc = _infer_hwc_channels(image)
        src_h, src_w, self.n_channels = self.source_hwc.shape

        self.root = tk.Tk()
        self.root.title(f"RNAscope ROI Drawer - {self.image_path.name}")
        self.root.minsize(1100, 760)

        max_w = max(700, self.root.winfo_screenwidth() - 360)
        max_h = max(500, self.root.winfo_screenheight() - 220)
        scale = min(max_w / src_w, max_h / src_h, 1.0)
        disp_w = max(1, int(round(src_w * scale)))
        disp_h = max(1, int(round(src_h * scale)))

        self.display = _DisplayContext(
            source_w=src_w,
            source_h=src_h,
            display_w=disp_w,
            display_h=disp_h,
        )
        self.preview_hwc = _resize_nearest_hwc(self.source_hwc, disp_h, disp_w)
        self.channel_names = _default_channel_names(self.n_channels)

        self.points: list[tuple[float, float]] = []
        self.line_item: int | None = None
        self.image_item: int | None = None
        self._photo: tk.PhotoImage | None = None
        self._ppm_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".ppm").name)

        self.channel_ui: list[_ChannelUi] = []

        self._build_layout()
        self._bind_events()
        self.update_display()

    def _build_layout(self) -> None:
        header = (
            f"Image: {self.image_path} | Output: {self.output_roi}\n"
            "Draw freehand with left mouse button. Use Save ROI when finished."
        )
        tk.Label(self.root, text=header, justify="left", anchor="w").pack(fill="x", padx=10, pady=(10, 6))

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=6)

        control_frame = tk.LabelFrame(body, text="Display Controls", padx=8, pady=8)
        control_frame.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(control_frame, text="Enable channels, choose LUTs, then set contrast percentiles.").pack(
            anchor="w", pady=(0, 6)
        )
        tk.Label(control_frame, text="Low/High range: 0-100").pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(control_frame)
        grid.pack(fill="x")
        tk.Label(grid, text="On", width=4).grid(row=0, column=0, sticky="w")
        tk.Label(grid, text="Channel", width=12).grid(row=0, column=1, sticky="w")
        tk.Label(grid, text="LUT", width=10).grid(row=0, column=2, sticky="w")
        tk.Label(grid, text="Low%", width=8).grid(row=0, column=3, sticky="w")
        tk.Label(grid, text="High%", width=8).grid(row=0, column=4, sticky="w")

        for i, name in enumerate(self.channel_names):
            enabled = tk.BooleanVar(value=True)
            lut = tk.StringVar(value=_default_lut_for_name(name))
            low_pct = tk.DoubleVar(value=1.0)
            high_pct = tk.DoubleVar(value=99.0)
            self.channel_ui.append(_ChannelUi(enabled=enabled, lut=lut, low_pct=low_pct, high_pct=high_pct))

            tk.Checkbutton(grid, variable=enabled, command=self.update_display).grid(row=i + 1, column=0, sticky="w")
            tk.Label(grid, text=name, width=12, anchor="w").grid(row=i + 1, column=1, sticky="w")
            tk.OptionMenu(grid, lut, *LUT_NAMES, command=lambda _v: self.update_display()).grid(
                row=i + 1, column=2, sticky="we"
            )
            tk.Entry(grid, textvariable=low_pct, width=8).grid(row=i + 1, column=3, sticky="w")
            tk.Entry(grid, textvariable=high_pct, width=8).grid(row=i + 1, column=4, sticky="w")

        controls_row = tk.Frame(control_frame)
        controls_row.pack(fill="x", pady=(10, 0))
        tk.Button(controls_row, text="Apply Display (R)", command=self.update_display).pack(side="left", padx=(0, 6))
        tk.Button(controls_row, text="Reset Contrast", command=self.reset_contrast).pack(side="left")

        spacer = tk.Frame(control_frame, height=8)
        spacer.pack(fill="x")

        draw_row = tk.Frame(control_frame)
        draw_row.pack(fill="x")
        tk.Button(draw_row, text="Clear ROI (C)", command=self.clear).pack(side="left", padx=(0, 6))
        tk.Button(draw_row, text="Save ROI (S)", command=self.save).pack(side="left", padx=(0, 6))
        tk.Button(draw_row, text="Quit (Esc)", command=self.quit).pack(side="left")

        canvas_frame = tk.Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.display.display_w,
            height=self.display.display_h,
            highlightthickness=1,
            highlightbackground="#888",
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(scrollregion=(0, 0, self.display.display_w, self.display.display_h))

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_motion)
        self.root.bind("<KeyPress-c>", lambda _: self.clear())
        self.root.bind("<KeyPress-C>", lambda _: self.clear())
        self.root.bind("<KeyPress-s>", lambda _: self.save())
        self.root.bind("<KeyPress-S>", lambda _: self.save())
        self.root.bind("<KeyPress-r>", lambda _: self.update_display())
        self.root.bind("<KeyPress-R>", lambda _: self.update_display())
        self.root.bind("<Escape>", lambda _: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _compose_rgb(self) -> np.ndarray:
        out = np.zeros((self.display.display_h, self.display.display_w, 3), dtype=np.float32)
        for i in range(self.n_channels):
            ui = self.channel_ui[i]
            if not ui.enabled.get():
                continue
            low = float(ui.low_pct.get())
            high = float(ui.high_pct.get())
            gray = _stretch_channel_to_uint8(self.preview_hwc[..., i], low, high)
            rgb = _apply_lut(gray, ui.lut.get()).astype(np.float32)
            out += rgb
        return np.clip(out, 0, 255).astype(np.uint8)

    def update_display(self) -> None:
        try:
            rgb = self._compose_rgb()
        except Exception as exc:
            messagebox.showerror("Display Error", f"Could not render channels:\n{exc}")
            return

        _write_ppm(rgb, self._ppm_path)
        self._photo = tk.PhotoImage(file=str(self._ppm_path))
        if self.image_item is None:
            self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfigure(self.image_item, image=self._photo)
        if self.line_item is not None:
            self.canvas.tag_raise(self.line_item)

    def reset_contrast(self) -> None:
        for ui in self.channel_ui:
            ui.low_pct.set(1.0)
            ui.high_pct.set(99.0)
        self.update_display()

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        x = min(max(x, 0.0), float(self.display.display_w - 1))
        y = min(max(y, 0.0), float(self.display.display_h - 1))
        return x, y

    def _start_draw(self, event: tk.Event) -> None:
        self.clear()
        x, y = self._clamp(event.x, event.y)
        self.points.append((x, y))
        self.line_item = self.canvas.create_line(x, y, x, y, fill="#00ffff", width=2, smooth=True)

    def _draw_motion(self, event: tk.Event) -> None:
        if self.line_item is None:
            return
        x, y = self._clamp(event.x, event.y)
        if self.points and abs(x - self.points[-1][0]) < 1 and abs(y - self.points[-1][1]) < 1:
            return
        self.points.append((x, y))

        if len(self.points) >= 2:
            first = self.points[0]
            coords = [*first]
            for px, py in self.points[1:]:
                coords.extend([px, py])
            coords.extend([first[0], first[1]])
            self.canvas.coords(self.line_item, *coords)

    def clear(self) -> None:
        self.points.clear()
        if self.line_item is not None:
            self.canvas.delete(self.line_item)
            self.line_item = None

    def _display_points_to_source(self) -> np.ndarray:
        pts = _clean_polyline(self.points)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts_xy = np.empty_like(pts, dtype=np.float64)
        pts_xy[:, 0] = pts[:, 0] * self.display.x_scale
        pts_xy[:, 1] = pts[:, 1] * self.display.y_scale
        return pts_xy

    def save(self) -> None:
        pts_xy = self._display_points_to_source()
        if pts_xy.shape[0] < 3:
            messagebox.showerror("ROI Too Small", "Draw a closed loop with at least 3 points before saving.")
            return
        try:
            _save_freehand_roi(
                pts_xy,
                self.output_roi,
                roi_name=self.roi_name,
                image_width=self.display.source_w,
                image_height=self.display.source_h,
            )
        except Exception as exc:  # pragma: no cover - GUI feedback path
            messagebox.showerror("Save Failed", str(exc))
            return

        messagebox.showinfo("Saved", f"Saved ROI to:\n{self.output_roi}")

    def quit(self) -> None:
        try:
            if self._ppm_path.exists():
                self._ppm_path.unlink()
        except OSError:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_roi_drawer(image_path: Path, output_roi: Path, roi_name: str | None = None) -> None:
    """Launch the ROI drawer GUI."""
    app = FreehandRoiApp(image_path=image_path, output_roi=output_roi, roi_name=roi_name)
    app.run()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw freehand ImageJ/Fiji ROI files with Tkinter.")
    parser.add_argument("--image", type=Path, required=True, help="Path to source image (typically .tif/.tiff).")
    parser.add_argument("--output-roi", type=Path, required=True, help="Output .roi file path.")
    parser.add_argument("--roi-name", type=str, default=None, help="ROI name stored in the ROI file header.")
    return parser


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = _build_arg_parser()
    args = parser.parse_args()
    launch_roi_drawer(image_path=args.image, output_roi=args.output_roi, roi_name=args.roi_name)


if __name__ == "__main__":  # pragma: no cover - direct execution
    main()
