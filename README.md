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
│   ├── video_viz.py                    # animate/grid/to_mp4/to_gif for video propagation results
│   ├── qc.py                           # post-hoc QC metrics + flagging for propagated mask stacks
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
├── single_object_depth_segmentation.py     # main script: single-cell video propagation pipeline
├── single_object_depth_segmentation_.ipynb # notebook version of the same pipeline
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

**`qc.py`** — post-hoc quality control for propagated mask stacks. `compute_metrics()` does a single-pass read of every `mask_NNNN.png` in a directory, computing per-frame signals (area, centroid, connected components, skeleton containment, predicted IoU) and frame-to-frame signals (area ratio, centroid jump, temporal IoU). Frames are flagged when any signal falls outside thresholds, and marked for manual intervention when two or more signals fire at once. `plot_traces()` shows the signal timeseries; `show_flagged()` renders a thumbnail strip of flagged frames with EM background. `save_masks()` handles writing `video_segments` dicts from propagation loops to disk as uint16 PNGs.

**`diagnostics.py`** — resource monitoring for long GPU sessions. `snapshot(label)` prints RAM, VRAM, disk usage, and open file handles in one call. `cleanup_vram()` runs `gc.collect()` + `torch.cuda.empty_cache()` then prints a snapshot. Works on Windows (kernel32 pagefile readout), Linux, and macOS.

**`diag_utils.py`** — an earlier standalone version of the diagnostics utilities, kept for backward compatibility with older notebooks.

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