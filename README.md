# segmentation-playground

A set of utilities for running [SAM2](https://github.com/facebookresearch/sam2) segmentation on electron microscopy (EM) image stacks from the Zhen Lab *C. elegans* connectome dataset. The core workflow: pull neuron skeleton annotations from CATMAID, align them to raw `.tif` stacks, feed those stacks into SAM2's image or video predictor, and QC the resulting mask propagations.

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
├── sam2_utils/                         # importable package
│   ├── __init__.py
│   ├── config.py                       # paths, SAM2 checkpoint registry, CATMAID settings, affine constants
│   ├── setup.py                        # device selection, checkpoint download, predictor construction
│   ├── catmaid.py                      # CATMAID REST client + fetch_all_annotations()
│   ├── alignment.py                    # CATMAID→tif affine: fit, apply, landmark picker helpers
│   ├── viz.py                          # show_mask/points/box, interactive point pickers
│   ├── video_viz.py                    # animate/grid/to_mp4/to_gif from in-memory video_segments
│   ├── review.py                       # read-only proofreading viewer: overlay saved masks from disk
│   ├── qc.py                           # QC metrics + flag/intervene rule (wired into pipeline.run_qc)
│   ├── diagnostics.py                  # RAM/VRAM/disk/file-handle snapshots, VRAM cleanup
│   ├── diag_utils.py                   # standalone diagnostics (Windows-only legacy, pre-package)
│   └── UTILS_README.md                 # module-level quick-start and usage examples
│
├── data/
│   ├── aggregate_data_pv.csv           # pre-fetched CATMAID skeleton nodes (stack-pixel coords)
│   ├── chains.json                     # z-slice chain definitions for video propagation
│   └── roots.json                      # skeleton root node lookup
│
├── checkpoints/                        # SAM2 model weights (downloaded by setup.ensure_checkpoint)
├── images/                             # sample EM crops for image-mode experiments
│
├── pipeline.py                             # phase functions + run_chain driver + ChainState (library)
├── run_aval.py                             # thin bootstrap: runs ONE chain (AVAL) through pipeline
├── single_object_depth_segmentation_.ipynb # reference notebook (what each phase does)
├── single_object_depth_segmentation.py     # SUPERSEDED by pipeline.py (kept for reference; stale)
├── multi_object_segmentation.ipynb         # multi-cell segmentation notebook
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## What each module does

**`config.py`** — central config for paths and constants. Set `WORM_PATH` to wherever your raw `.tif` stack lives. Holds the SAM2 checkpoint registry (tiny/small/base_plus/large), CATMAID URL and project ID, and the pre-fit affine matrix (`M_AFFINE`, `T_AFFINE`) that maps CATMAID stack-pixel coordinates to tif-pixel coordinates.

**`setup.py`** — one-time session setup. `build_predictor(size, kind)` picks the right device (CUDA > MPS > CPU), enters bfloat16 autocast on CUDA, downloads the checkpoint if it's missing, and returns a predictor ready to use. Call it once per notebook session.

**`catmaid.py`** — thin REST wrapper around the Zhen Lab CATMAID instance. `fetch_all_annotations()` pulls every skeleton's node coordinates into a single DataFrame, converting from nanometers to stack-pixel coordinates using `config.STACK_RESOLUTION_NM`.

**`alignment.py`** — registers CATMAID coordinates to tif-image coordinates via a least-squares affine fit. The stored `M_AFFINE`/`T_AFFINE` in `config.py` was fit from 12 landmarks at CATMAID z=1293. `fit_affine(landmarks)` lets you refit from a new landmark set; `catmaid_to_tif(x, y)` applies the stored transform. `sample_nodes_grid` helps pick evenly-spread landmark candidates when collecting a new fit.

**`viz.py`** — display helpers. `show_mask`, `show_points`, `show_box`, `show_masks` are ports of Meta's notebook helpers, unified to work the same way in both image and video predictor contexts. `pick_point` and `pick_landmark` are interactive matplotlib widget pickers for collecting prompt coordinates or affine landmarks in a notebook (requires `%matplotlib widget`).

**`video_viz.py`** — visualizes SAM2 video-mode propagation results. `animate()` builds an inline scrubber/player (returns IPython HTML); `grid()` produces a static N-frame thumbnail grid; `to_mp4()` and `to_gif()` write to disk. All three read directly from the JPEG frames SAM2 wrote to disk during propagation, so masks and frames stay in sync without any coordinate math.

**`review.py`** — read-only proofreading viewer for a *finished* chain. Where `video_viz` overlays the in-memory `video_segments` a run just produced, `review` rebuilds the same overlay from the saved artifacts on disk (`masks/`, `state.json`, `qc.csv`) and delegates rendering back to `video_viz`. `animate()` / `grid()` show the whole chain; `grid_flagged()` / `animate_flagged()` show only the QC-flagged frames; `to_gif()` / `to_mp4()` export. Strictly read-only by design — it is the proofreading tool, **not** the M4 intervention GUI. Reuses `qc._iter_mask_paths` / `qc._load_binary` so mask reading has a single definition.

**`qc.py`** — quality control for propagated mask stacks; the auto-detection core. `compute_metrics()` does a single-pass read of every `mask_<z>.png` in a directory, computing per-frame signals (area, centroid, connected components, skeleton containment) and frame-to-frame signals (area ratio, centroid jump, temporal IoU). `skeleton_contained` is tri-state — `True` / `False` / `NaN` (no chain node at that z; not assessable) — and only an explicit `False` flags. A frame is flagged when any signal fires and marked `intervene` when ≥2 fire; thresholds are parameters (defaults preserve the original rule). `plot_traces()` shows the signal timeseries; `show_flagged()` renders a thumbnail strip of flagged frames with EM background. `save_masks()` writes `video_segments` dicts to disk as uint16 *instance-label* PNGs (a multi-object concern; single-object runs use `pipeline.save_masks`'s 0/255 format instead). `qc` is wired into the pipeline via `pipeline.run_qc`, which runs it over each finished chain headlessly.

**`diagnostics.py`** — resource monitoring for long GPU sessions. `snapshot(label)` prints RAM, VRAM, disk usage, and open file handles in one call. `cleanup_vram()` runs `gc.collect()` + `torch.cuda.empty_cache()` then prints a snapshot. Works on Windows (kernel32 pagefile readout), Linux, and macOS.

**`diag_utils.py`** — an earlier standalone version of the diagnostics utilities, kept for backward compatibility with older notebooks.

**`pipeline.py`** — the notebook lifted into a library of phase functions (`select_anchor`, `load_frame_sam`, `build_prompts`, `image_predict`, `box_from_mask`, `prepare_video_frames`, `propagate`, `save_masks`, `run_qc`) plus a thin `run_chain` driver that threads a serializable `ChainState` through all nine steps. Per-run tunables live on `PipelineConfig` (resolution, prompts, QC thresholds); static project facts stay in `config.py`. `run_chain` reproduces the notebook's AVAL masks pixel-for-pixel and now ends with a QC + flagging step that writes `qc.csv` and sets the chain's `status`. `state.json` persists everything for resume / re-open. Importing `pipeline` is light (no torch/cv2 at import); heavy deps load lazily inside the phases.

**`run_aval.py`** — the bootstrap driver: builds the predictors, loads the CSV/CATMAID annotations and `chains.json`, and runs ONE chain (AVAL) through `pipeline.run_chain`. It is the only place that knows about predictors and the filesystem layout, and it is the M1 regression harness. Run `python run_aval.py` (running `pipeline.py` directly does nothing, by design).

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

---

## Notes

- The affine constants in `config.py` were fit at CATMAID z=1293. If you refit on a different section, update `M_AFFINE` and `T_AFFINE` there so all notebooks pick up the new values.
- `setup_device()` enters a bfloat16 autocast context that persists for the session. Calling it more than once is safe.
- `qc.save_masks()` expects `frame_to_z` to map SAM2 frame indices to CATMAID z values. Build it from your tif filename list before the propagation loop.
- The current end-to-end run path is `python run_aval.py` (one chain through `pipeline.run_chain`), not the standalone `single_object_depth_segmentation.py`, which is superseded by `pipeline.py`.
- Each run writes `output/<neuron>/chain_NN/` with `masks/`, `state.json`, and `qc.csv`. To proofread, open it read-only with `review.animate(chain_dir)` / `review.grid(chain_dir)` in a notebook, or `review.grid_flagged(chain_dir)` / `review.to_gif(chain_dir, out)` (the inline `animate` only renders in a notebook and hits matplotlib's embed limit on long chains — use `grid_flagged`/`to_gif` from a script).
- QC thresholds are `qc_*` knobs on `PipelineConfig` (e.g. `qc_temporal_iou_min`, `qc_area_ratio_bounds`, `qc_skeleton_dilation_px`); tune them there, not in `qc.py`.