# Code map: where to change what

Use this table to find the file that owns a concern. The rule the codebase follows is "one
concern, one home": each thing lives in exactly one place, so a change to do one thing touches
as few files as possible.

## I want to change...

| Goal | Edit | Notes |
|------|------|-------|
| A segmentation phase | the matching `pipeline/` submodule | The core is a package split by concern: `predict.py` (anchor select, prompt build, image predict, multimask select, box, anchor gate, tier-1 crop), `propagate.py` (video propagation + per-slice re-seed `segment_per_slice`), `crop.py` (tier-2 windows + frame prep), `frames.py` (FrameStore + EM loading), `masks.py` (save + postprocess), `qc.py` (per-chain QC), `orchestrator.py` (`run_chain`). Phase functions take plain arrays and a config, do no I/O setup. |
| A run/tuning knob (scale, crop, QC thresholds, seed shape) | `PipelineConfig` in `pipeline/config.py` | One dataclass holds every tunable. Defaults reproduce the original single-chain run. (Distinct from `sam2_utils/config.py`, which holds static paths/constants.) |
| The per-chain state object or its serialization | `pipeline/state.py` | `ChainState`, `Prompts`, `AnchorScore`, and the `state_to_dict`/`from_dict` + `save_state`/`load_state` round-trip. |
| The default knobs for a named run (which worm, model, paths, tier-2) | `sam2_utils/presets.py` | `--preset original` (target worm) and `--preset eval` (cross-worm GT). Any CLI flag overrides. |
| A filesystem path, model checkpoint, affine constant, or CATMAID setting | `sam2_utils/config.py` | Central constants. Import-light: no torch or cv2 at module load. |
| Any coordinate transform (`_tif` / `_sam` / `_crop` / `_pcrop` / `_cm`, z maps, nm to px, crop windows) | `sam2_utils/alignment.py` | The single home for space conversions. Do not write `/ scale` inline anywhere else. |
| How the headless batch runs, resumes, or builds the triage queue | `batch.py` | The headless driver. Builds predictors once, runs every chain, writes the manifest. `--backend {sam2,sam3}` (default `sam2`, unchanged) plus `--sam3-checkpoint PATH` route through `build_predictors(cfg)` to the SAM3 adapters instead of `setup.build_predictor`; see the sam3_backend.py row below and [../how-to/run-sam3-on-narval.md](../how-to/run-sam3-on-narval.md) for the cluster run. |
| The review GUI (layers, keys, correction tools) | `gui.py` | The napari driver, per-CHAIN paradigm. Composes `review`, `review_queue`, `labels`, and `pipeline`. |
| The neuron-level review GUI (whole neuron on one crop canvas) | `gui_neuron.py` | The second paradigm: opens a whole neuron, branches as labels in one Labels layer on a per-neuron `_ncrop` crop. Imports shared pieces from `gui.py`. See the 2026-06-23 spec/plan under `docs/superpowers/`. |
| The work queue or review-status ledger the GUI reads and writes | `sam2_utils/review_queue.py` | Owns `_review.csv`, separate from the batch's `_manifest.csv`. |
| The per-frame label store (the training data the GUI collects) | `sam2_utils/labels.py` | One flat row per labelled frame in `_labels.csv`. Pure pandas. |
| A QC signal or its threshold | `sam2_utils/qc.py` + the `qc_*` knobs on `PipelineConfig` | Metrics live in `qc.py`; thresholds live on the config so a run tunes them in one place. |
| How a finished chain is loaded and visualized (read-only) | `sam2_utils/review.py` | Rebuilds the overlay from a chain's on-disk artifacts. Not the correction GUI. |
| Predictor construction, device selection, checkpoint download | `sam2_utils/setup.py` | Builds the image or video predictor. Imports torch lazily. |
| SAM3 as a SAM2 drop-in (model-swap backend) | `sam2_utils/sam3_backend.py` | Thin adapters (`Sam3ImagePredictor`, `Sam3VideoPredictor`) presenting the SAM2 predictor interface over HuggingFace SAM3 PVS trackers, so `segment_per_slice` and `propagate` run SAM3 unchanged, plus torch-free prompt/output helpers. torch and transformers import lazily. The 2x2 SAM2-vs-SAM3 comparison driver is `experiments/sam3_bakeoff.py`; findings in [../explanation/sam3-bakeoff-findings.md](../explanation/sam3-bakeoff-findings.md). |
| The CATMAID client or annotation fetch | `sam2_utils/catmaid.py` | REST wrapper plus `fetch_all_annotations`. |
| RAM/VRAM/disk diagnostics for long runs | `sam2_utils/diagnostics.py` | Snapshots and VRAM cleanup. torch is lazy here too. |
| Ground-truth evaluation (region IoU, VOI, ARAND, ERL, registration) | `eval/` | `score_batch.py`, `metrics.py`, `erl.py`, `registration.py`, `gt_dataset.py`. The GT-free target-worm bleed / dropout scorer is `merge_metric.py`: it owns `MembraneSource` (the EM-patch loader that feeds the membrane pass) and the membrane-aware scoring, on top of the Phase-0 foreign-skeleton-node containment. `concat_merge_shards.py` stitches a sharded `merge_metric` run (from `cluster/run_eval_array.sh`) into the canonical `_merge_metric.csv`; `retro_eval.py` builds one comparison table across run trees (merge-metric plus compute and QC columns). |
| The per-pixel membrane signal and its detector primitives (bleed/underfill detection against the raw EM) | `sam2_utils/membrane.py` | `membrane_map` (v1 dark-ridge filter, swappable interface for a trained model later) plus the pure detector scalars `spanning_membrane`, `boundary_on_membrane`, `underfill_fraction`. Library-side (not `eval/`) because the deferred grow-to-membrane refinement and non-overlap arbitration reuse this same signal from `pipeline/`. See [ADR 0016](../adr/0016-membrane-map-border-to-border-bleed-detection.md). |
| The single-chain regression run | `run_aval.py` | Runs one chain end to end. The worked example and reproduction harness. |
| Running the batch in parallel on the Narval cluster | `cluster/` | Slurm array over `batch.py`: `make_chunks.py` (neurons to chunks), `run_array.sh` (the array job, forwards `VENV`/`PRESET`/`SAM_BACKEND`/`SAM3_CKPT`/`OUT_ROOT`/`CHUNKS` so it can drive either backend), `run_exp.sh` + `run_merge_exp.sh` (the resolution-experiment presets), `merge_shards.py` + `run_merge.sh` (stitch segmentation shards), `stage_download.sh` (tar `*_merged` trees for pull-down). Sharded scoring: `run_eval_array.sh` (split one merged tree's merge-metric across CPU tasks into per-shard CSVs) then `eval.concat_merge_shards` to stitch them. `run_py.sh` runs any CPU step (merge, concat, retro-eval) as a dependency job off the login node; `submit_sam3_round.sh` queues the whole SAM3 test round in one shot with `afterok` dependencies. See [../how-to/run-on-narval.md](../how-to/run-on-narval.md), [../how-to/run-sam3-on-narval.md](../how-to/run-sam3-on-narval.md), and [../how-to/queued-sam3-narval-tests.md](../how-to/queued-sam3-narval-tests.md). |

## The four entry points

| Run this | What it does |
|----------|--------------|
| `py -3 run_aval.py` | One chain, headless. The smallest end-to-end run. Start here. |
| `py -3 batch.py --preset original --neurons AVAL` | The headless batch over selected chains, with resume. Add `--backend sam3 --sam3-checkpoint PATH` to run the same chains through SAM3 instead of SAM2 (default `sam2`, unchanged). |
| `py -3 gui.py` | The napari review and correction GUI on flagged chains. |
| `py -3 -m eval.score_batch --preset eval` | Score predictions against the cross-worm ground truth. |

## The two output trees

The filesystem is the database. A run writes two separate trees. See
[state-and-storage.md](state-and-storage.md) for the full schema.

- `output/` holds per-chain masks, state, and QC, plus the cross-chain manifest and triage CSVs.
- `frames_root/` holds the SAM2 JPEG frames: a shared decode cache and per-chain link views.

## Dependency direction

The library does not import the drivers. `pipeline.py` and `sam2_utils/` import only each other
and third-party packages. The drivers (`batch.py`, `gui.py`, `gui_neuron.py`, `run_aval.py`), `eval/`, and the `cluster/`
scripts import the library, never the reverse. (`gui_neuron.py` also imports `gui.py`, which is
driver to driver, allowed.) If you add an import that points from the core out to a driver, you have
introduced a cycle. See [../explanation/architecture.md](../explanation/architecture.md).
