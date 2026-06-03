# segmentation-playground

A set of utilities for running [SAM2](https://github.com/facebookresearch/sam2) segmentation on electron microscopy (EM) image stacks from the Zhen Lab *C. elegans* connectome dataset. The core workflow: pull neuron skeleton annotations from CATMAID, align them to raw `.tif` stacks, feed those stacks into SAM2's image or video predictor, and QC the resulting mask propagations.

The codebase is mid-transition from the original exploratory notebook into a semi-automatic, human-in-the-loop pipeline (a `sam2_utils` helper package + a `pipeline.py` library + thin drivers). The design principle is **automation-first, triage-second**: auto-run and auto-QC every chain, and spend the human only on flagged frames. Everything runs locally on one box — filesystem only, no server or database. See `PIPELINE_CONTEXT.md` for architecture, the milestone roadmap, and known coordinate/filename/mask-space gotchas.

---

## Setup

```bash
git clone https://github.com/farrellsid/segmentation-playground
cd segmentation-playground
pip install -r requirements.txt
```

SAM2 itself isn't on PyPI, install it separately from Meta's repo:

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Or let `setup.check_sam2_available(auto_install=True)` do it for you at runtime.

**CATMAID token:** create a `.env` file at the repo root:

```
CATMAID_TOKEN=your_token_here
```

No quotes, no spaces. The token is read from `$CATMAID_TOKEN` env var first, then this file as a fallback. Don't commit it.

---

## File structure

```
segmentation-playground/
├── sam2_utils/                         # importable package (stateless helpers)
│   ├── __init__.py
│   ├── config.py                       # paths, SAM2 checkpoint registry, CATMAID settings, affine constants
│   ├── setup.py                        # device selection, checkpoint download, predictor construction
│   ├── catmaid.py                      # CATMAID REST client + fetch_all_annotations()
│   ├── alignment.py                    # CATMAID→tif affine: fit, apply, landmark picker helpers
│   ├── viz.py                          # show_mask/points/box, interactive point pickers
│   ├── video_viz.py                    # animate/grid/to_mp4/to_gif for in-RAM propagation results
│   ├── review.py                       # read-only proofreading viewer for finished chains on disk
│   ├── qc.py                           # QC metrics + flagging signals for propagated mask stacks
│   ├── diagnostics.py                  # RAM/VRAM/disk/file-handle snapshots, VRAM cleanup
│   ├── diag_utils.py                   # standalone diagnostics (Windows-only legacy, pre-package)
│   └── UTILS_README.md                 # module-level quick-start and usage examples
│
├── pipeline.py                         # phase functions + ChainState + run_chain (the library)
├── run_aval.py                         # single-chain bootstrap driver (regression harness)
├── batch.py                            # headless batch runner + resume + triage queue
│
├── data/
│   ├── aggregate_data_pv.csv           # pre-fetched CATMAID skeleton nodes (stack-pixel coords)
│   ├── chains.json                     # z-slice chain definitions for video propagation
│   └── roots.json                      # skeleton root node lookup
│
├── checkpoints/                        # SAM2 model weights (downloaded by setup.ensure_checkpoint)
├── images/                             # sample EM crops for image-mode experiments
│
├── single_object_depth_segmentation_.ipynb # source-of-truth notebook (single-cell pipeline)
├── multi_object_segmentation.ipynb         # multi-cell segmentation notebook
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Run outputs

`batch.py` / `run_chain` write a filesystem-indexed tree (no server, no DB):

```
output/
  _manifest.csv               # every chain × status (pending/running/done/flagged) — drives resume
  _triage.csv                 # flagged frames across all chains — feeds the review tool / GUI
  _timing.csv                 # per-chain runtime telemetry
  <neuron>/chain_<idx:02d>/
    state.json                # serialized ChainState (resume / re-open without recompute)
    qc.csv                    # per-frame QC metrics, indexed by catmaid_z
    masks/mask_<catmaid_z:04d>.png
```

---

## What each module does

**`config.py`** — central config for paths and constants. Set `WORM_PATH` to wherever your raw `.tif` stack lives. Holds the SAM2 checkpoint registry (tiny/small/base_plus/large), CATMAID URL and project ID, and the pre-fit affine matrix (`M_AFFINE`, `T_AFFINE`) that maps CATMAID stack-pixel coordinates to tif-pixel coordinates.

**`setup.py`** — one-time session setup. `build_predictor(size, kind)` picks the right device (CUDA > MPS > CPU), enters bfloat16 autocast on CUDA, downloads the checkpoint if it's missing, and returns a predictor ready to use. Call it once per notebook session.

**`catmaid.py`** — thin REST wrapper around the Zhen Lab CATMAID instance. `fetch_all_annotations()` pulls every skeleton's node coordinates into a single DataFrame, converting from nanometers to stack-pixel coordinates using `config.STACK_RESOLUTION_NM`.

**`alignment.py`** — registers CATMAID coordinates to tif-image coordinates via a least-squares affine fit. The stored `M_AFFINE`/`T_AFFINE` in `config.py` was fit from 12 landmarks at CATMAID z=1293. `fit_affine(landmarks)` lets you refit from a new landmark set; `catmaid_to_tif(x, y)` applies the stored transform. `sample_nodes_grid` helps pick evenly-spread landmark candidates when collecting a new fit.

**`viz.py`** — display helpers. `show_mask`, `show_points`, `show_box`, `show_masks` are ports of Meta's notebook helpers, unified to work the same way in both image and video predictor contexts. `pick_point` and `pick_landmark` are interactive matplotlib widget pickers for collecting prompt coordinates or affine landmarks in a notebook (requires `%matplotlib widget`).

**`video_viz.py`** — visualizes SAM2 video-mode propagation results. `animate()` builds an inline scrubber/player (returns IPython HTML); `grid()` produces a static N-frame thumbnail grid; `to_mp4()` and `to_gif()` write to disk. All three read directly from the JPEG frames SAM2 wrote to disk during propagation, so masks and frames stay in sync without any coordinate math.

**`review.py`** — read-only proofreading viewer for a *finished* chain on disk. A sibling of `video_viz`: where `video_viz` overlays the in-RAM `video_segments` dict, `review` rebuilds the same overlay from a chain's saved `state.json` + `masks/` + `qc.csv`, so you can re-open and proofread long after the run without re-running SAM2. Deliberately read-only — it is **not** the correction GUI (no point editing, no re-prompting; that is the single napari tool in milestone 4). It reuses `video_viz`'s rendering and `qc`'s mask-reading helpers so "how a mask is read" has one definition across the package.

**`qc.py`** — post-hoc quality control for propagated mask stacks. `compute_metrics()` does a single-pass read of every `mask_NNNN.png` in a directory, computing per-frame signals (area, centroid, connected components, skeleton containment, predicted IoU) and frame-to-frame signals (area ratio, centroid jump, temporal IoU). Frames are flagged when any signal falls outside thresholds, and marked for manual intervention when two or more signals fire at once. `plot_traces()` shows the signal timeseries; `show_flagged()` renders a thumbnail strip of flagged frames with EM background; `export_triage()` writes the flagged-frame rows that `batch.py` rolls up into the cross-chain `_triage.csv`. (`save_masks()` lives here for notebook use, but the canonical writer the pipeline calls is `pipeline.save_masks()`.)

**`diagnostics.py`** — resource monitoring for long GPU sessions. `snapshot(label)` prints RAM, VRAM, disk usage, and open file handles in one call. `cleanup_vram()` runs `gc.collect()` + `torch.cuda.empty_cache()` then prints a snapshot. Works on Windows (kernel32 pagefile readout), Linux, and macOS.

**`diag_utils.py`** — an earlier standalone version of the diagnostics utilities, kept for backward compatibility with older notebooks.

---

## Pipeline & drivers

These top-level scripts turn the `sam2_utils` helpers into the actual segmentation pipeline. The package stays stateless; these own the state, the filesystem layout, and the data sources.

**`pipeline.py`** — the library the notebook lifts into (milestone 1). Holds the phase functions (`select_anchor`, `load_frame_sam`, `build_prompts`, `image_predict`, `box_from_mask`, `prepare_video_frames`, `propagate`, `save_masks`, `run_qc`), the `PipelineConfig`/`Prompts`/`ChainState` dataclasses, `save_state`/`load_state` (→ `state.json`), and `run_chain` — the thin driver that runs one chain end-to-end (anchor → image predict → box → propagate → save → QC). Coordinate spaces are tagged by suffix (`_cm`/`_tif`/`_sam`) and z by name (`catmaid_z`/`file_z`/`frame_idx`); masks are stored at `_sam` (`save_downscale == scale`) as `mask_<catmaid_z:04d>.png`. `python pipeline.py` does nothing by design — it's a library.

**`run_aval.py`** — single-chain bootstrap driver and regression harness. Runs one chain (AVAL) through `run_chain` and serializes the `ChainState`; its masks should match the notebook's output pixel-for-pixel. The one place that knows about predictors, the CSV/CATMAID source, `chains.json`, and the output paths — edit the knobs at the top to match your box. `python run_aval.py`.

**`batch.py`** — headless batch runner + resume (milestone 3). `run_aval.py` generalized into a loop: build the session once, then run *every* chain unattended, recording status to `_manifest.csv` as it goes and rolling per-chain QC flags into one cross-chain `_triage.csv`. Survives crashes (resume from the manifest), never recomputes a finished chain, and writes `_timing.csv` telemetry. Treats each chain as a single atomic `run_chain` — mid-propagation halt-and-re-prompt belongs to the milestone-4 napari GUI, not here. `python batch.py`.

---

## Quick start for notebooks

```python
from sam2_utils import setup, viz, diagnostics, config
from pathlib import Path
import cv2, torch, numpy as np

# 1. Build predictor (downloads checkpoint if needed)
predictor, device = setup.build_predictor(size="small", kind="image")
diagnostics.snapshot("after model load")

# 2. Load a tif slice
tif_files = sorted(config.WORM_PATH.glob("*.tif"))
image = cv2.cvtColor(cv2.imread(str(tif_files[0])), cv2.COLOR_BGR2RGB)

# 3. Prompt and predict
with torch.inference_mode():
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[4550, 4990]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

# 4. Display and clean up
order = np.argsort(scores)[::-1]
viz.show_masks(image, masks[order], scores[order],
               point_coords=np.array([[4550, 4990]]),
               input_labels=np.array([1]))
predictor.reset_predictor()
diagnostics.cleanup_vram()
```

For video propagation and multi-object workflows, see the notebooks and `UTILS_README.md`.

### Running the pipeline (non-notebook)

```bash
python run_aval.py     # one chain end-to-end (regression harness); edit knobs at top
python batch.py        # all chains, unattended, with resume + _manifest.csv + _triage.csv
```

Both reuse the same `pipeline.run_chain`. `batch.py` is the primary mode: it auto-runs and auto-QCs every chain and surfaces only flagged frames (`_triage.csv`) for human review — open a finished chain with `sam2_utils.review` to proofread. Resume is automatic: re-running `batch.py` skips chains already `done`/`flagged` in the manifest.

---

## Notes

- The affine constants in `config.py` were fit at CATMAID z=1293. If you refit on a different section, update `M_AFFINE` and `T_AFFINE` there so all notebooks pick up the new values. Should be accurate enough but just in case.
- `setup.build_predictor()` (which calls `setup.setup_device()`) enters a bfloat16 autocast context that persists for the session. Calling it more than once is safe.
- `pipeline.save_masks()` / `qc` expect `frame_to_z` to map SAM2 frame indices to CATMAID z values. The pipeline builds this from the tif filename list before propagation; masks are written as `mask_<catmaid_z:04d>.png` (no `z` prefix). Old `mask_z<...>.png` files from the notebook are skipped — re-run through the pipeline.
- Milestone status lives in `PIPELINE_CONTEXT.md §6`. As of June 2026: M1 (library/state-machine), M2 (inline QC + flagging), and M3 (headless batch runner) are done; M4 is the single napari review/correction GUI; M5 is per-neuron aggregation → Blender.