# RNAscope End‑to‑End Analysis Pipeline

A modular Python pipeline for analyzing 3‑channel RNAscope images using polygon tissue ROIs and point‑ROI maxima.
The workflow:

1. Loads a single **top‑level** full image per region (e.g., *hippo*, *thal*) for each animal folder (e.g., `Rat1`).
2. Reads **polygon** ROIs (e.g., `DG.roi`, `CA1.roi`) and rasterizes them to masks.
3. Reads **point** ROIs (maxima) per region from `./maxima` for **GOA** and **GOB**.
4. For each polygon ROI: crops the bounding box, **free‑form masks** the crop (outside polygon → 0), draws maxima crosses, segments nuclei from masked DAPI (Cellpose), expands labels by ≈2.5 µm, and computes spot counts & metrics.
5. Writes QC images and CSV tables: per‑nucleus, per‑ROI, and overall.

---

## Requirements

* **Python** ≥ 3.9 (3.10–3.12 recommended)
* Packages:

  * `cellpose`
  * `scikit-image`
  * `numpy`
  * `pandas`
  * `tifffile`
  * `roifile`

Install in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install cellpose scikit-image numpy pandas tifffile roifile
```

> **GPU (optional):** Cellpose will use GPU if available (CUDA/CuDNN properly installed). Otherwise it falls back to CPU.

---

## Expected Directory Layout

```
project_root/
├─ Rat1/
│  ├─ <top-level image containing "hippo">.tif
│  ├─ hippo_rois/           # polygon ROIs (e.g., DG.roi, CA1.roi)
│  ├─ <top-level image containing "thal">.tif
│  └─ thal_rois/
├─ Rat2/
│  └─ ...
└─ maxima/
   ├─ Rat1_hippo_GOA_maxima.roi
   ├─ Rat1_hippo_GOB_maxima.roi
   ├─ Rat1_thal_GOA_maxima.roi
   └─ Rat1_thal_GOB_maxima.roi
```

* **Top‑level images only** are used per `Rat*` folder (no recursion). The filename must contain the region keyword (e.g., `hippo`, `thal`).
* **Polygon ROIs** go in `hippo_rois/` or `thal_rois/` and are applied to the matching region image.
* **Point ROIs** (maxima) per region live in `./maxima`. Flexible naming: any tag containing `goa` or `gob` is detected; canonical example: `Rat3_hippo_GOA_maxima.roi`.

---

## Image & ROI Assumptions

* RNAscope images are 3‑channel TIFFs (DAPI, GoB, GoA). The pipeline auto‑detects layout:

  * **HWC**: `(H, W, C)` when last axis has 3/4
  * **CHW**: `(C, H, W)` when first axis has 3/4
* DAPI/GoB/GoA **channel indices** are configurable.
* Polygon ROIs are ImageJ `.roi` with valid polygon coordinates.
* Maxima ROIs are ImageJ **point** ROIs for GOA & GOB in global image coordinates.

---

## Outputs

All results are written under `results_rnascope/` (configurable):

* **`cutouts/`** – Free‑form masked ROI cutouts (outside polygon set to 0). One per polygon ROI.
* **`qc_overlays/`** – Normalised GOA and GOB channel overlays with their respective maxima crosses (GOB=red, GOA=cyan) saved as PNG.
* **`roi_masks_cropped/`** – Cropped polygon masks for sanity checks (binary 0/255).
* **`masks/`** – Cellpose **labels** per polygon ROI (cached for reuse).
* **`csv/`** – Tables:

  * `per_nucleus_counts.csv`
  * `per_roi_summary.csv`
  * `overall_metrics.csv`

---

## How It Works (Pipeline)

For each animal `RatX` and region `hippo`/`thal` (only if the corresponding ROI folder exists):

1. **Full Image** – Load the *top‑level* region image and infer layout (CHW/HWC).
2. **Maxima** – Load GOA/GOB point ROIs for the region from `./maxima`.
3. **Per Polygon ROI**

   * Rasterize polygon to mask (auto XY‑swap if needed).
   * Crop the bounding box and create a **free‑form masked** cutout (outside polygon → 0).
   * Save the masked cutout and a masked‑DAPI overlay with maxima crosses.
   * Segment nuclei from **masked** DAPI with Cellpose (labels cached to disk).
   * Expand labels by ≈2.5 µm (configurable), then **clip expansion to polygon**.
   * Map maxima to nucleus (nucleus, expanded, ring) and accumulate metrics.
4. **Tables** – Write per‑nucleus, per‑ROI, and overall CSVs.

---

## Metrics

Per **nucleus**:

* Areas (µm²): nucleus, expanded, ring (expanded − nucleus)
* Spot counts: GoA/GoB in nucleus, expanded, ring
* Flags: `more_GoA_than_GoB`, `more_GoB_than_GoA`, `any_GoA`, `any_GoB`, `no_spots_both`
* **Grading bins** (per expanded counts):

  * Bin 0 = 0
  * Bin 1 = 1–3
  * Bin 2 = 4–9
  * Bin 3 = 10–15
  * Bin 5 = >15

Per **ROI**:

* ROI area (µm²)
* Totals & densities (spots/µm²) for GoA & GoB
* GOA:GOB ratio (NaN if GOB=0)
* Average spots per cell for GoA & GoB (cell-associated only) and their ratio
* Proportions: cells with more GoA/Gob, any GoA/GoB, or no spots
* Bin counts & proportions; **H‑score** = Σ(bin × count) and normalized per cell

**Overall**: same summaries pooled across all nuclei, including average spots per cell metrics.

---

## Configuration

Configuration is encapsulated in a `Config` dataclass. Edit the values in your own driver or when invoking `run_pipeline`.

| Field              | Type    | Default            | Description                                          |
| ------------------ | ------- | ------------------ | ---------------------------------------------------- |
| `root`             | `Path`  | `.`                | Project root containing `Rat*` folders and `maxima/` |
| `out_dir`          | `Path`  | `results_rnascope` | Output directory root                                |
| `pixel_size_um`    | `float` | `0.1455`           | Microscopy pixel size (µm/pixel)                     |
| `expansion_um`     | `float` | `2.5`              | Label expansion distance (µm)                        |
| `dapi_index`       | `int`   | `0`                | DAPI channel index                                   |
| `gob_index`        | `int`   | `1`                | GoB channel index                                    |
| `goa_index`        | `int`   | `2`                | GoA channel index                                    |
| `spot_marker_size` | `int`   | `3`                | Cross half‑size for maxima overlay                   |
| `load_saved_masks` | `bool`  | `True`             | Reuse Cellpose labels if present                     |
| `transpose_xy`     | `bool`  | `True`             | Swap ROI (x,y) → (y,x) if needed                     |
| `skip_empty_roi`   | `bool`  | `True`             | Skip ROIs that rasterize empty after auto‑fix        |
| `debug`            | `bool`  | `True`             | Verbose progress printing                            |

> **CHW/HWC:** The pipeline auto‑detects channel layout and applies masks on the correct axis. No manual conversion required.

---

## Running the Pipeline

1. Ensure the directory structure matches **Expected Directory Layout** above.
2. Run the pipeline using the package:

```bash
python -m rnascope_pipeline
```

You should see logs like:

```
=== RNAscope pipeline | root=/path/to/project | expand=2.5µm≈17px ===
found 3 experiment(s): ['Rat1', 'Rat2', 'Rat3']
— Processing Rat1 —
  [Rat1/hippo] image='20250804_rat1_hippo_63x.tif' shape=(3, 22184, 18025) layout=CHW
  [Rat1/hippo] maxima: GOA=Rat1_hippo_GOA_maxima.roi, GOB=Rat1_hippo_GOB_maxima.roi
  [Rat1/hippo] 4 ROI(s): ['CA1', 'CA3', 'DG', 'DGfun']
  > ROI 'CA1': starting analysis
    cutout bbox=(6024:11257,3824:5297) size=(3, 5233, 1473) masked_size=(3, 5233, 1473)
    maxima counts in ROI: GOA=27842, GOB=11821
    saved GOA overlay → results_rnascope/qc_overlays/Rat1_hippo_CA1_goa_maxima.png
    saved GOB overlay → results_rnascope/qc_overlays/Rat1_hippo_CA1_gob_maxima.png
    computed & saved labels: n_nuclei=...
    expanded labels by 17px (~2.5 µm)
    ROI area=... µm² | densities: GOA=.../µm², GOB=.../µm² | ratio GOA:GOB=...
  < ROI 'CA1': done in 47.6s
```

---

## Troubleshooting

**“Expected exactly one top‑level image … found N”**
Make sure only **one** file directly under each `Rat*` folder contains the region keyword (e.g., `hippo`). Move duplicates into subfolders or rename them.

**“ROI produced an empty mask …”**
The polygon’s coordinates may be in the opposite order (x,y vs y,x), from a different base image, or malformed. The pipeline auto‑tries both XY conventions. If still empty, check:

* ROI header bounds vs image size
* Set `transpose_xy=True` (default) or flip it if needed
* Inspect the saved cropped mask in `roi_masks_cropped/`

**Nuclei = 0**
Most often due to off‑tissue ROI, incorrect channel layout, or weak DAPI. We normalize DAPI (1–99%) and run on **masked** DAPI; verify the overlay image looks reasonable. If needed, try a different Cellpose model or add light preprocessing.

**CHW/HWC confusion**
The pipeline logs the detected layout. Masking, overlays, and segmentation all respect that layout. No action required.

**Performance**
Enable a GPU for Cellpose if possible. Large numbers of ROIs or very large images can be compute‑intensive.

---

## Extending

* Swap nuclei segmentation model (e.g., StarDist) by replacing `segment_nuclei` in `segmentation.py`.
* Add per‑cell intensity features by sampling `cutout_masked` with nucleus labels.
* Export spots as per‑ROI CSVs with local/global coordinates.

---

## Acknowledgements

* **Cellpose**: Stringer et al. ([https://www.cellpose.org/](https://www.cellpose.org/))
* **scikit‑image, numpy, pandas, tifffile, roifile** for core IO & image ops.

---

## License

Add your project’s license terms here.
