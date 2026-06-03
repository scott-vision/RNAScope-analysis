"""Persistent dataset storage and snapshot/run bookkeeping for CPSAM fine-tuning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Iterable
from uuid import uuid4

import numpy as np
import tifffile as tiff
import yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceEntry:
    source_id: str
    path: str
    active: bool
    default_include: bool
    added_at: str
    notes: str


MANIFEST_FIELDS = [
    "patch_id",
    "img_path",
    "mask_path",
    "source_id",
    "source_image_path",
    "source_image_hash",
    "x",
    "y",
    "w",
    "h",
    "created_at",
    "seed",
    "generator_version",
    "annotation_status",
    "include",
    "split",
    "last_modified_at",
    "annotator",
    "notes",
]


class DatasetStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.sources_path = workspace / "sources.json"
        self.manifest_path = workspace / "manifest.csv"
        self.patches_images_dir = workspace / "patches" / "images"
        self.patches_masks_dir = workspace / "patches" / "masks"
        self.snapshots_dir = workspace / "snapshots"
        self.runs_dir = workspace / "runs"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.patches_images_dir.mkdir(parents=True, exist_ok=True)
        self.patches_masks_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.sources_path.exists():
            self.sources_path.write_text("[]\n", encoding="utf-8")
        if not self.manifest_path.exists():
            with self.manifest_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()

    def load_sources(self) -> list[SourceEntry]:
        try:
            data = json.loads(self.sources_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = []
        out: list[SourceEntry] = []
        for row in data:
            out.append(
                SourceEntry(
                    source_id=str(row.get("source_id", "")),
                    path=str(row.get("path", "")),
                    active=bool(row.get("active", True)),
                    default_include=bool(row.get("default_include", True)),
                    added_at=str(row.get("added_at", _utc_now_iso())),
                    notes=str(row.get("notes", "")),
                )
            )
        return out

    def save_sources(self, sources: list[SourceEntry]) -> None:
        payload = [
            {
                "source_id": s.source_id,
                "path": s.path,
                "active": s.active,
                "default_include": s.default_include,
                "added_at": s.added_at,
                "notes": s.notes,
            }
            for s in sources
        ]
        self.sources_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert_source(self, path: Path, *, active: bool = True, default_include: bool = True) -> SourceEntry:
        abs_path = str(path.expanduser().resolve())
        sources = self.load_sources()
        for s in sources:
            if Path(s.path) == Path(abs_path):
                s.active = active
                s.default_include = default_include
                self.save_sources(sources)
                return s
        entry = SourceEntry(
            source_id=f"src_{uuid4().hex[:8]}",
            path=abs_path,
            active=active,
            default_include=default_include,
            added_at=_utc_now_iso(),
            notes="",
        )
        sources.append(entry)
        self.save_sources(sources)
        return entry

    def remove_source(self, source_id: str) -> None:
        sources = [s for s in self.load_sources() if s.source_id != source_id]
        self.save_sources(sources)

    def read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def write_manifest(self, rows: list[dict[str, str]]) -> None:
        with self.manifest_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})

    def _next_patch_id(self) -> str:
        rows = self.read_manifest()
        max_n = 0
        for r in rows:
            pid = str(r.get("patch_id", ""))
            if pid.startswith("patch_"):
                tail = pid.split("_", 1)[1]
                if tail.isdigit():
                    max_n = max(max_n, int(tail))
        return f"patch_{max_n + 1:06d}"

    def add_patch(
        self,
        *,
        patch_img_path: Path,
        patch_mask_path: Path,
        source_id: str,
        source_image_path: Path,
        source_image_hash: str,
        x: int,
        y: int,
        w: int,
        h: int,
        seed: int,
        annotation_status: str = "auto",
        include: bool = True,
        split: str = "unassigned",
        annotator: str = "",
        notes: str = "",
    ) -> str:
        patch_id = self._next_patch_id()
        img_out = self.patches_images_dir / f"{patch_id}_img.tif"
        mask_out = self.patches_masks_dir / f"{patch_id}_masks.tif"
        shutil.copy2(patch_img_path, img_out)
        shutil.copy2(patch_mask_path, mask_out)

        created = _utc_now_iso()
        row = {
            "patch_id": patch_id,
            "img_path": str(img_out.resolve()),
            "mask_path": str(mask_out.resolve()),
            "source_id": source_id,
            "source_image_path": str(source_image_path.resolve()),
            "source_image_hash": source_image_hash,
            "x": str(x),
            "y": str(y),
            "w": str(w),
            "h": str(h),
            "created_at": created,
            "seed": str(seed),
            "generator_version": "1",
            "annotation_status": annotation_status,
            "include": "true" if include else "false",
            "split": split,
            "last_modified_at": created,
            "annotator": annotator,
            "notes": notes,
        }
        rows = self.read_manifest()
        rows.append(row)
        self.write_manifest(rows)
        return patch_id

    def update_patch_fields(self, patch_id: str, **fields: str) -> None:
        rows = self.read_manifest()
        changed = False
        for r in rows:
            if r.get("patch_id") != patch_id:
                continue
            for k, v in fields.items():
                if k in MANIFEST_FIELDS:
                    r[k] = str(v)
            r["last_modified_at"] = _utc_now_iso()
            changed = True
            break
        if changed:
            self.write_manifest(rows)

    def remove_patch(self, patch_id: str) -> bool:
        rows = self.read_manifest()
        kept: list[dict[str, str]] = []
        removed_row: dict[str, str] | None = None
        for r in rows:
            if r.get("patch_id") == patch_id and removed_row is None:
                removed_row = r
                continue
            kept.append(r)
        if removed_row is None:
            return False
        self.write_manifest(kept)
        for k in ("img_path", "mask_path"):
            p = Path(removed_row.get(k, ""))
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        return True

    def list_patch_rows(
        self,
        *,
        source_ids: Iterable[str] | None = None,
        approved_only: bool = False,
        include_only: bool = False,
    ) -> list[dict[str, str]]:
        rows = self.read_manifest()
        out: list[dict[str, str]] = []
        allow_sources = set(source_ids) if source_ids is not None else None
        for r in rows:
            if allow_sources is not None and r.get("source_id", "") not in allow_sources:
                continue
            if approved_only and r.get("annotation_status", "") != "approved":
                continue
            if include_only and r.get("include", "").lower() != "true":
                continue
            out.append(r)
        return out

    def count_instance_masks(self, mask_path: Path) -> int:
        arr = tiff.imread(str(mask_path))
        return max(0, len({int(x) for x in arr.ravel().tolist() if int(x) > 0}))

    def build_snapshot(
        self,
        *,
        name: str,
        source_ids: list[str] | None,
        approved_only: bool,
        include_only: bool,
        split_ratios: tuple[float, float, float],
        stratify_by_source: bool,
        seed: int,
        nuclei_channel_index: int = 0,
    ) -> str:
        snapshot_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        snap_dir = self.snapshots_dir / snapshot_id
        train_dir = snap_dir / "train"
        val_dir = snap_dir / "val"
        test_dir = snap_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot export selection is include-driven.
        # Keep approved_only parameter for backward compatibility, but ignore it.
        rows = self.list_patch_rows(
            source_ids=source_ids, approved_only=False, include_only=include_only
        )
        if len(rows) == 0:
            raise ValueError("No patches matched snapshot filters. Mark patches include=true.")
        rng = random.Random(seed)
        ratios = split_ratios

        def assign_split(items: list[dict[str, str]]) -> None:
            rng.shuffle(items)
            n = len(items)
            n_train = int(round(n * ratios[0]))
            n_val = int(round(n * ratios[1]))
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            for i, it in enumerate(items):
                if i < n_train:
                    it["split"] = "train"
                elif i < n_train + n_val:
                    it["split"] = "val"
                else:
                    it["split"] = "test"

        valid_splits = {"train", "val", "test"}
        has_manual_splits = any((r.get("split", "").strip().lower() in valid_splits) for r in rows)
        all_split_assigned = all((r.get("split", "").strip().lower() in valid_splits) for r in rows)
        if has_manual_splits and all_split_assigned:
            for r in rows:
                r["split"] = r.get("split", "train").strip().lower()
        else:
            if stratify_by_source:
                by_source: dict[str, list[dict[str, str]]] = {}
                for r in rows:
                    by_source.setdefault(r.get("source_id", "unknown"), []).append(r)
                for vals in by_source.values():
                    assign_split(vals)
            else:
                assign_split(rows)

        for r in rows:
            pid = r["patch_id"]
            split = r.get("split", "train")
            target_root = train_dir if split == "train" else val_dir if split == "val" else test_dir
            img_src = Path(r["img_path"])
            mask_src = Path(r["mask_path"])
            img = tiff.imread(str(img_src))
            nuc = self._extract_nuclei_channel(img, nuclei_channel_index)
            tiff.imwrite(str(target_root / f"{pid}_img.tif"), nuc)
            shutil.copy2(mask_src, target_root / f"{pid}_masks.tif")

        snapshot_yaml = {
            "snapshot_id": snapshot_id,
            "name": name,
            "created_at": _utc_now_iso(),
            "source_filters": source_ids,
            "rules": {
                "approved_only": False,
                "include_only": include_only,
                "stratify_by_source": stratify_by_source,
                "seed": seed,
                "nuclei_channel_index": nuclei_channel_index,
            },
            "split_policy": {
                "train": ratios[0],
                "val": ratios[1],
                "test": ratios[2],
            },
            "patch_ids": [r["patch_id"] for r in rows],
            "counts": {
                "train": len(list((train_dir).glob("*_img.tif"))),
                "val": len(list((val_dir).glob("*_img.tif"))),
                "test": len(list((test_dir).glob("*_img.tif"))),
            },
            "export_paths": {
                "train": str(train_dir.resolve()),
                "val": str(val_dir.resolve()),
                "test": str(test_dir.resolve()),
            },
        }
        with (snap_dir / "snapshot.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(snapshot_yaml, fh, sort_keys=False)
        return snapshot_id

    @staticmethod
    def _extract_nuclei_channel(image: np.ndarray, idx: int) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4) and idx < arr.shape[-1]:
                return arr[..., idx]
            if arr.shape[0] in (3, 4) and idx < arr.shape[0]:
                return arr[idx, ...]
            # Generic fallback for atypical channel counts.
            if idx < arr.shape[-1]:
                return arr[..., idx]
            if idx < arr.shape[0]:
                return arr[idx, ...]
        raise ValueError(f"Cannot extract nuclei channel {idx} from shape {arr.shape}")

    def create_run(self, *, snapshot_id: str, train_command: str, params: dict[str, str]) -> str:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "started_at": _utc_now_iso(),
            "ended_at": "",
            "snapshot_id": snapshot_id,
            "train_command": train_command,
            "params": params,
            "status": "running",
            "model_output_path": "",
            "log_path": str((run_dir / "train.log").resolve()),
            "metrics_path": "",
        }
        with (run_dir / "run.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        return run_id

    def finalize_run(
        self, run_id: str, *, status: str, model_output_path: str = "", metrics_path: str = ""
    ) -> None:
        run_path = self.runs_dir / run_id / "run.yaml"
        if not run_path.exists():
            return
        with run_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        payload["ended_at"] = _utc_now_iso()
        payload["status"] = status
        payload["model_output_path"] = model_output_path
        payload["metrics_path"] = metrics_path
        with run_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    def list_runs(self) -> list[dict]:
        out: list[dict] = []
        for run_yaml in sorted(self.runs_dir.glob("run_*/run.yaml")):
            try:
                with run_yaml.open("r", encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh) or {}
                out.append(payload)
            except OSError:
                continue
        return out

    def list_snapshots(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for snap_dir in sorted(self.snapshots_dir.glob("snap_*")):
            if not snap_dir.is_dir():
                continue
            snapshot_id = snap_dir.name
            name = ""
            created_at = ""
            snapshot_yaml = snap_dir / "snapshot.yaml"
            if snapshot_yaml.exists():
                try:
                    with snapshot_yaml.open("r", encoding="utf-8") as fh:
                        payload = yaml.safe_load(fh) or {}
                    name = str(payload.get("name", "")).strip()
                    created_at = str(payload.get("created_at", "")).strip()
                except OSError:
                    pass
            out.append(
                {
                    "snapshot_id": snapshot_id,
                    "name": name,
                    "created_at": created_at,
                }
            )
        return out

    def delete_snapshot(self, snapshot_id: str) -> bool:
        snapshot_id = str(snapshot_id).strip()
        if not snapshot_id or Path(snapshot_id).name != snapshot_id or not snapshot_id.startswith("snap_"):
            return False
        snap_dir = self.snapshots_dir / snapshot_id
        if not snap_dir.is_dir():
            return False
        shutil.rmtree(snap_dir)
        return True
