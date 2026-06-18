# Code map: where to change what

Use this table to find the file that owns a concern. The rule the codebase follows is "one
concern, one home": each thing lives in exactly one place, so a change to do one thing touches
as few files as possible.

## I want to change...

| Goal | Edit | Notes |
|------|------|-------|
| A segmentation phase | the matching `pipeline/` submodule | The core is a package split by concern: `predict.py` (anchor select, prompt build, image predict, multimask select, box, anchor gate, tier-1 crop), `propagate.py` (video propagation), `crop.py` (tier-2 windows + frame prep), `frames.py` (FrameStore + EM loading), `masks.py` (save + postprocess), `qc.py` (per-chain QC), `orchestrator.py` (`run_chain`). Phase functions take plain arrays and a config, do no I/O setup. |
| A run/tuning knob (scale, crop, QC thresholds, seed shape) | `PipelineConfig` in `pipeline/config.py` | One dataclass holds every tunable. Defaults reproduce the original single-chain run. (Distinct from `sam2_utils/config.py`, which holds static paths/constants.) |
| The per-chain state object or its serialization | `pipeline/state.py` | `ChainState`, `Prompts`, `AnchorScore`, and the `state_to_dict`/`from_dict` + `save_state`/`load_state` round-trip. |
| The default knobs for a named run (which worm, model, paths, tier-2) | `sam2_utils/presets.py` | `--preset original` (target worm) and `--preset eval` (cross-worm GT). Any CLI flag overrides. |
| A filesystem path, model checkpoint, affine constant, or CATMAID setting | `sam2_utils/config.py` | Central constants. Import-light: no torch or cv2 at module load. |
| Any coordinate transform (`_tif` / `_sam` / `_crop` / `_pcrop` / `_cm`, z maps, nm to px, crop windows) | `sam2_utils/alignment.py` | The single home for space conversions. Do not write `/ scale` inline anywhere else. |
| How the headless batch runs, resumes, or builds the triage queue | `batch.py` | The headless driver. Builds predictors once, runs every chain, writes the manifest. |
| The review GUI (layers, keys, correction tools) | `gui.py` | The napari driver. Composes `review`, `review_queue`, `labels`, and `pipeline`. |
| The work queue or review-status ledger the GUI reads and writes | `sam2_utils/review_queue.py` | Owns `_review.csv`, separate from the batch's `_manifest.csv`. |
| The per-frame label store (the training data the GUI collects) | `sam2_utils/labels.py` | One flat row per labelled frame in `_labels.csv`. Pure pandas. |
| A QC signal or its threshold | `sam2_utils/qc.py` + the `qc_*` knobs on `PipelineConfig` | Metrics live in `qc.py`; thresholds live on the config so a run tunes them in one place. |
| How a finished chain is loaded and visualized (read-only) | `sam2_utils/review.py` | Rebuilds the overlay from a chain's on-disk artifacts. Not the correction GUI. |
| Predictor construction, device selection, checkpoint download | `sam2_utils/setup.py` | Builds the image or video predictor. Imports torch lazily. |
| The CATMAID client or annotation fetch | `sam2_utils/catmaid.py` | REST wrapper plus `fetch_all_annotations`. |
| RAM/VRAM/disk diagnostics for long runs | `sam2_utils/diagnostics.py` | Snapshots and VRAM cleanup. torch is lazy here too. |
| Ground-truth evaluation (region IoU, VOI, ARAND, ERL, registration) | `eval/` | `score_batch.py`, `metrics.py`, `erl.py`, `registration.py`, `gt_dataset.py`. |
| The single-chain regression run | `run_aval.py` | Runs one chain end to end. The worked example and reproduction harness. |

## The four entry points

| Run this | What it does |
|----------|--------------|
| `py -3 run_aval.py` | One chain, headless. The smallest end-to-end run. Start here. |
| `py -3 batch.py --preset original --neurons AVAL` | The headless batch over selected chains, with resume. |
| `py -3 gui.py` | The napari review and correction GUI on flagged chains. |
| `py -3 -m eval.score_batch --preset eval` | Score predictions against the cross-worm ground truth. |

## The two output trees

The filesystem is the database. A run writes two separate trees. See
[state-and-storage.md](state-and-storage.md) for the full schema.

- `output/` holds per-chain masks, state, and QC, plus the cross-chain manifest and triage CSVs.
- `frames_root/` holds the SAM2 JPEG frames: a shared decode cache and per-chain link views.

## Dependency direction

The library does not import the drivers. `pipeline.py` and `sam2_utils/` import only each other
and third-party packages. The drivers (`batch.py`, `gui.py`, `run_aval.py`) and `eval/` import the
library, never the reverse. If you add an import that points from the core out to a driver, you have
introduced a cycle. See [../explanation/architecture.md](../explanation/architecture.md).
