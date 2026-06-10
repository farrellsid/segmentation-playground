# segmentation-playground

Tooling for running [SAM2](https://github.com/facebookresearch/sam2) segmentation on electron-microscopy (EM) image stacks from the Zhen Lab *C. elegans* connectome dataset. The job: segment ~300 neurons (a few thousand maximal-linear-chains) out of a ~300-slice `.tif` stack into per-neuron mask volumes for export to Blender.

The core workflow: pull neuron skeleton annotations from CATMAID, align them to the raw `.tif` stack, seed SAM2 on an anchor frame, propagate the mask through z with SAM2's video predictor, and QC the result. Everything runs locally on one Windows + GPU box, filesystem only, no server or database.

The codebase is mid-transition from a single notebook into a **semi-automatic, human-in-the-loop pipeline** (design principle: *automation-first, triage-second* - auto-run and auto-QC every chain, surface only flagged frames to a human). For the architecture, roadmap, and design rationale, see **`PIPELINE_CONTEXT.md`** - it is the big-picture reference. This README covers structure, modules, and how to run things.

---

## Setup

```bash
git clone https://github.com/farrellsid/segmentation-playground
cd segmentation-playground
pip install -r requirements.txt
```

SAM2 itself isn't on PyPI; install it separately from Meta's repo:

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Or let `setup.check_sam2_available(auto_install=True)` do it for you at runtime.

**CATMAID token:** create a `.env` file at the repo root:

```
CATMAID_TOKEN=your_token_here
```

No quotes, no spaces. The token is read from `$CATMAID_TOKEN` first, then this file as a fallback. Don't commit it.

**Paths:** all the per-run filesystem locations live in one place, `sam2_utils/config.py`: point `WORM_PATH` at your raw `.tif` stack, and set `OUTPUT_ROOT` / `FRAMES_ROOT` (the mask-output and JPEG-scratch roots) for your box. The in-repo data sources (`CSV_PATH`, `CHAINS_PATH`, `ROOTS_PATH`) are derived from the repo location, so they resolve automatically. The driver scripts (`run_aval.py`, `batch.py`) import these from `config`, so you edit them once, in `config.py`.

---

## File structure

```
segmentation-playground/
├── sam2_utils/                         # importable helper package (stable utilities)
│   ├── __init__.py
│   ├── config.py                       # paths (data + output), SAM2 checkpoint registry, CATMAID settings, affine constants
│   ├── setup.py                        # device selection, checkpoint download, predictor construction
│   ├── catmaid.py                      # CATMAID REST client + fetch_all_annotations()
│   ├── alignment.py                    # THE coordinate-transform home (affine, tif<->sam, z maps, nm->px, CropWindow)
│   ├── viz.py                          # show_mask/points/box, interactive point pickers (notebook)
│   ├── video_viz.py                    # animate/grid/to_mp4/to_gif for propagation results (notebook)
│   ├── qc.py                           # QC metrics + flagging over a saved mask stack
│   ├── review.py                       # read-only proofreading viewer for a finished chain
│   ├── labels.py                       # M4: per-frame label store (the "label engine") - torch/napari-free
│   ├── review_queue.py                 # M4: work queue + GUI-owned review-status ledger - torch/napari-free
│   ├── diagnostics.py                  # RAM/VRAM/disk snapshots, peak-VRAM probes, VRAM cleanup
│   └── UTILS_README.md                 # module-level quick-start and usage snippets
│
├── pipeline.py                         # the pipeline LIBRARY: PipelineConfig, ChainState, phase
│                                       #   functions, run_chain, PropagationSession, run_qc
├── run_aval.py                         # thin driver: run ONE chain (the M1 regression harness)
├── batch.py                            # thin driver: headless batch over all chains + manifest/resume
├── gui.py                              # thin driver: napari review/triage/correction GUI (M4)
│
├── tests/
│   ├── test_alignment.py               # torch-free unit tests for the coordinate transforms
│   ├── test_anchor_select.py           # torch-free tests for the multimask anchor auto-select
│   ├── test_labels.py                  # torch-free tests for the M4 label store
│   └── test_review_queue.py            # torch-free tests for the M4 queue + review ledger
│
├── data/                              # (git-ignored) CATMAID-derived inputs + GT landing spot
│   ├── aggregate_data_pv.csv           # pre-fetched CATMAID skeleton nodes (stack-pixel coords)
│   ├── chains.json                     # per-neuron maximal-linear-chain definitions
│   ├── roots.json                      # skeleton root node lookup
│   └── groundtruth/                    # cross-worm GT + matching EM (Stage 0/2; README tracked, data ignored)
│
├── experiments/                        # A/B harnesses + sweeps, kept for repro (see experiments/README.md)
│   ├── ab_fallback.py ab_seed.py ab_tier2.py ab_tier2_wide.py ab_underfill.py
│   ├── sweep_dilation.py
│   ├── ab_figs/                        # A/B comparison PNGs (disposable, flagged)
│   └── *.log                           # captured run logs (disposable; git-ignored going forward)
├── notebooks/                          # reference notebooks (see notebooks/README.md)
│   ├── single_object_depth_segmentation_.ipynb  # source-of-truth notebook: one chain, end-to-end
│   └── multi_object_segmentation.ipynb          # multi-cell segmentation notebook
├── archive/                            # off-critical-path material (see archive/README.md)
│   ├── make_deck_figures.ipynb                  # deck/report figure generation
│   └── figures/                                 # the deck/report figure outputs
├── eval/                               # NEXT PHASE scaffold: Stage 0 eval harness (ERL+VOI) — stub only
├── finetune/                           # NEXT PHASE scaffold: Stage 2 SAM2 finetuning — stub only
│   └── finetune.py                              # placeholder
│
├── checkpoints/                        # (git-ignored) SAM2 model weights (downloaded by setup.ensure_checkpoint)
├── references/                         # (git-ignored) research PDFs behind FUTURE_DIRECTIONS
├── output/                             # (git-ignored) generated: per-chain masks/state/qc + cross-chain manifest (see below)
│
├── PIPELINE_CONTEXT.md                 # lean current-state reference (read this first)
├── PIPELINE_HISTORY.md                 # the build log: milestones, closed-issue resolutions, A/B results, decisions
├── FUTURE_DIRECTIONS.md                # research-informed roadmap (Stage 0 -> Stage 4)
├── GUI_GUIDE.md                        # M4 napari GUI user guide (layers, keys, workflows)
├── pyproject.toml
├── requirements.txt
└── README.md
```

> **Repo layout (June 2026 tidy).** The durable library (`pipeline.py`, `sam2_utils/`, the drivers,
> `tests/`, `data/`) is separated from scratch: A/B harnesses live in `experiments/`, reference
> notebooks in `notebooks/`, shelved code in `archive/`, and `eval/` + `finetune/` are empty
> next-phase scaffolds. See `PIPELINE_CONTEXT.md` §8 item 32 for the keep/archive/delete
> needs-decision checklist.

> The old `single_object_depth_segmentation.py` script and `sam2_utils/diag_utils.py` are gone. The notebook -> library refactor (milestone 1) replaced the script with `pipeline.py` + `run_aval.py`, and `diag_utils.py` was an earlier standalone diagnostics module superseded by `diagnostics.py`. The `_.ipynb` notebook remains the reference for *what* each phase does; `pipeline.py` is the source of truth for *how* it runs.

---

## How it works

The refactor's shape is **library + thin drivers**, not "notebook -> one big script":

- **`pipeline.py` is a pure library.** It holds the phase functions (`select_anchor`, `build_prompts`, `image_predict` / `anchor_crop_predict`, `box_from_mask`, `prepare_video_frames`, `propagate`, `postprocess_mask`, `save_masks`, `run_qc`), a serializable per-chain `ChainState`, a single `PipelineConfig` of knobs, and `run_chain` - the driver that runs one chain through all phases and writes its artifacts. It imports no predictors and does no I/O setup of its own; `python pipeline.py` does nothing by design.
- **Thin drivers call the library.** `run_aval.py` runs a single chain (and is the regression harness; its masks match the notebook pixel-for-pixel). `batch.py` builds the predictors once and runs every chain headless, recording status to a manifest and rolling flagged frames into a triage queue. The notebooks are for exploration.
- **State is checkpointed per chain.** Each chain serializes to `state.json`, so a run can be paused, resumed after a crash, or re-opened later without recomputation.
- **QC runs inline.** After saving a chain's masks, `run_chain` scores them (`run_qc`) and sets the chain's status to `done` or `flagged`, all headless. Only flagged frames reach a human.

The napari review GUI (milestone 4) is now in: `gui.py` is a fourth thin driver that reads the batch's flagged chains and drives the same library — `PropagationSession` for re-segmentation, `pipeline.run_qc` for re-scoring — to let a human correct and re-propagate, logging every decision as a training label. The learned predictor model trained on those labels (milestone 4.5) and Blender export (milestone 5) are not built yet; see `PIPELINE_CONTEXT.md` §6 for the roadmap and §"M4 status" below for what M4 ships vs. defers.

---

## What each module does

**`config.py`** - central constants. `WORM_PATH` (raw `.tif` stack), the data/output paths (`DATA_DIR`, `CSV_PATH`, `CHAINS_PATH`, `ROOTS_PATH`, `OUTPUT_ROOT`, `FRAMES_ROOT` - the one home the drivers import), the SAM2 checkpoint registry (tiny/small/base_plus/large), CATMAID URL and project ID, `STACK_RESOLUTION_NM`, `FILE_Z_OFFSET`, and the pre-fit affine (`M_AFFINE`, `T_AFFINE`) that maps CATMAID stack-pixel coordinates to tif-pixel coordinates. (Note: the *runtime* pipeline knobs - scale, QC thresholds, crop, etc. - live on `PipelineConfig` in `pipeline.py`, not here.)

**`setup.py`** - one-time session setup. `build_predictor(size, kind)` picks the device (CUDA > MPS > CPU), enters bfloat16 autocast on CUDA, downloads the checkpoint if missing, and returns `(predictor, device)` for `kind="image"` or `kind="video"`. `setup_device()`, `ensure_checkpoint()`, and `check_sam2_available()` are the lower-level pieces. torch is imported lazily, so importing the package doesn't require it.

**`catmaid.py`** - thin REST wrapper around the Zhen Lab CATMAID instance. `fetch_all_annotations()` pulls every skeleton's nodes into one DataFrame, converting nanometers to stack-pixel coordinates via `alignment.nm_to_stack_px` (which reads `config.STACK_RESOLUTION_NM`).

**`alignment.py`** - the single home for every coordinate transform (PIPELINE_CONTEXT §4). The spaces are `_cm` (CATMAID stack px), `_tif` (full-res), `_sam` (= `_tif` / scale; the SAM2 video input and the canonical on-disk mask space), and `_crop` (the high-res anchor crop), plus the z section maps and the nm voxel divide. Functions: the CATMAID->tif affine (`fit_affine` refits from a landmark set, `catmaid_to_tif` applies the stored transform fit from 12 landmarks at CATMAID z=1293, `sample_nodes_grid` picks spread-out landmark candidates); the resolution maps `tif_to_sam` / `sam_to_tif`; the z maps `catmaid_z_to_file_z` / `file_z_to_catmaid_z`; `nm_to_stack_px`; and **`CropWindow`**, where the `_tif <-> _crop <-> _sam` crop math lives, with the only row/col `[y,x]` swap isolated to `CropWindow.slice_tif`. `tests/test_alignment.py` guards all of this.

**`viz.py`** - notebook display helpers. `show_mask`, `show_points`, `show_box`, `show_masks` (ports of Meta's notebook helpers); `pick_point` / `pick_landmark` are interactive matplotlib-widget pickers (need `%matplotlib widget`).

**`video_viz.py`** - visualizes propagation results during exploration. `animate()` (inline scrubber), `grid()` (thumbnail grid), `to_mp4()` / `to_gif()` (write to disk), reading directly from the JPEG frames SAM2 wrote so masks and frames stay in sync.

**`pipeline.py`** - the pipeline library (see *How it works*). `PipelineConfig` is the single tuning surface; `ChainState` is the serializable per-chain record (now including per-phase timing); `run_chain` runs one chain through every phase; `PropagationSession` is the interruptible propagation primitive (seed -> lazily yield per-frame results -> break -> add points/mask -> resume over the same `inference_state`); `run_qc` scores a saved chain and writes `qc.csv`. `pipeline.save_masks` writes masks as **0/255 uint8 single-channel** PNGs (the notebook's format, directly viewable and pixel-comparable).

**`qc.py`** - quality control over a saved mask stack. `compute_metrics()` does a single-pass read of every `mask_NNNN.png` in a directory, computing per-frame signals (area, centroid, connected components, skeleton containment, predicted IoU) and frame-to-frame signals (area ratio, centroid jump, temporal IoU). A frame is *flagged* when any signal trips its threshold and *intervene* when two or more fire at once; thresholds are `qc_*` knobs on `PipelineConfig` (tune there, not in `qc.py`). `plot_traces()` / `show_flagged()` are diagnostics; `export_triage()` writes the flagged-frame CSV. (`qc.save_masks()` writes uint16 *instance-label* PNGs - a multi-object concern reserved for later; the single-object pipeline uses `pipeline.save_masks` instead.)

**`review.py`** - read-only proofreading viewer. `load_chain()` rebuilds the overlay from a finished chain's on-disk artifacts (`masks/`, `state.json`, `qc.csv`) and delegates rendering to `video_viz`; `animate` / `grid` show the whole chain, `animate_flagged` / `grid_flagged` show only QC-flagged frames, `to_mp4` / `to_gif` export. Strictly read-only - it is *not* the (not-yet-built) correction GUI.

**`diagnostics.py`** - resource monitoring for long GPU runs. `snapshot(label)` prints RAM, VRAM, disk, and open file handles; `cleanup_vram()` runs GC + `torch.cuda.empty_cache()`; `reset_peak_vram()` / `peak_vram_gb()` bracket a run for the batch driver's per-chain VRAM telemetry. torch is imported lazily, so the module (and `import sam2_utils`) works on a torch-free box. Works on Windows, Linux, macOS.

**`labels.py`** (M4) - the GUI's per-frame **label store** ("label engine"). One flat row per labelled frame in `output/_labels.csv`: the QC signal vector (features), the human verdict + error_type, the chain's anchor verdict (the §7 anchor-contamination guard), the frame's role, and whether the rule flagged it. `LabelStore.record()` upserts idempotently per `(neuron, chain_idx, z)`; `sample_unflagged()` logs a uniform random sample of *un-flagged* frames (the §7 selection-bias guard — the only window onto silent errors). Pure pandas, torch/napari-free, so the schema is unit-tested (`tests/test_labels.py`). M4 *collects* these labels; training a model on them is M4.5.

**`review_queue.py`** (M4) - the GUI's **work queue + review-status ledger**. Reads the batch's `_manifest.csv` (flagged chains) and `_triage.csv` (per-frame detail) read-only, and owns a *separate* `output/_review.csv` with its own status column (`unreviewed` → `in_review` → `approved`/`rejected`/`corrected`). Keeping review status in its own file is the cheap form of the §7 "partition ownership" requirement: the batch and the GUI never write the same column. `pending()` is flagged-minus-disposed (re-surfacing `in_review` chains a crashed session left behind); `claim()`/`set_status()` upsert dispositions; `refresh()` re-reads the manifest to pick up chains a still-running batch flagged. Pure pandas, torch/napari-free, unit-tested (`tests/test_review_queue.py`).

**`gui.py`** (M4) - the napari **review/triage/correction GUI**, the one human-facing tool. A thin driver that composes `review` (rebuild a chain's overlay from disk), `review_queue` (which chains need a human), `labels` (log decisions), and `pipeline` (`image_predict` / `box_from_mask` / `PropagationSession` / `run_qc` for re-segmentation). Two-tier loading keeps GPU off the critical path: the *light* tier (annotate_df + chains + on-disk artifacts) is enough to browse, scrub, inspect flags, paint, and label; the SAM2 predictors are built **lazily** only when the human triggers a re-run/resume. Everything it shows lives in one `_sam` grid (EM frames, masks, points), so a click is already an `_sam` coordinate. See the quick-start below and the §"M4 status" notes for what it ships vs. defers.

---

## Storage layout

The filesystem *is* the database. A run writes two trees:

```
output/
  _manifest.csv               # every chain x status - drives the batch run + resume
  _triage.csv                 # flagged (intervene) frames across all chains
  _timing.csv                 # per-chain timing + peak VRAM
  <neuron>/
    chain_00/
      state.json              # the serialized ChainState
      qc.csv                  # per-frame QC metrics (+ a `queue` column)
      masks/mask_<z:04d>.png   # 0/255 uint8, canonical _sam space (save_downscale == scale)
    chain_01/ ...

frames_root/                  # SAM2 JPEG frames - a separate tree from output/
  frames_cache_s<scale>/
    z<file_z>.jpg             # shared decode cache: each EM frame downscaled once, ever
  chain_views/
    <neuron>_chain<idx>_s<scale>/
      00000.jpg ...           # 0-indexed links into the cache, per chain
```

Resume is automatic: a chain already `done`/`flagged` is skipped on re-run; an interrupted chain is left `running` and retried next launch.

---

## Quick start

### Run one chain (headless)

`run_aval.py` is the worked example - edit the path/cell knobs at the top (paths come from `config`), then `python run_aval.py`. The essence:

```python
import json, pandas as pd
from pathlib import Path
from sam2_utils import setup, alignment, diagnostics, config
from pipeline import PipelineConfig, ChainState, run_chain, save_state

cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                     output_root=config.OUTPUT_ROOT, frames_root=config.FRAMES_ROOT)

# annotations: cached CSV (or catmaid.fetch_all_annotations(...)), then apply the affine
annotate_df = pd.read_csv(config.CSV_PATH)
xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]

# pick a chain
chains = json.load(open(config.CHAINS_PATH))
chain = [c for c in chains if c["cell_name"] == "AVAL"][2]

# build both predictors once (heavy on VRAM at size="large")
image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image")
video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video")

state = ChainState(neuron="AVAL", chain_idx=2, config=cfg)
state = run_chain(state, image_predictor=image_predictor, video_predictor=video_predictor,
                  annotate_df=annotate_df, chain=chain,
                  on_video_phase=diagnostics.cleanup_vram)
save_state(state, Path(cfg.output_root) / "AVAL" / "chain_02" / "state.json")
print(state.status)   # 'done' or 'flagged'
```

Then proofread it read-only:

```python
from sam2_utils import review
chain_dir = Path(config.OUTPUT_ROOT) / "AVAL" / "chain_02"
review.grid_flagged(chain_dir)                       # only QC-flagged frames
review.to_gif(chain_dir, chain_dir / "aval.gif")
```

### Run the whole dataset (headless batch)

Edit the knobs at the top of `batch.py` (`NEURONS` allow-list or `None` for all, `CLEAN`, `GIF_MODE`; the path roots come from `config`), then:

```bash
python batch.py
```

It builds the predictors once, runs every selected chain through `run_chain`, and writes `output/_manifest.csv`, `_triage.csv`, and `_timing.csv`. Re-running resumes where it left off. `clean=True` wipes prior outputs first (scope-aware: full reset when `neurons=None`, else just the named neurons).

### Review & correct flagged chains (the M4 napari GUI)

> **Full walkthrough: [`GUI_GUIDE.md`](GUI_GUIDE.md)** — layers, every button/key, workflows,
> "next CHAIN vs next flagged FRAME", and the low-res explanation. The summary below is the gist.

After a batch run, `output/_manifest.csv` has the chains QC flagged. Open the GUI to clear them:

```bash
pip install napari magicgui            # GUI-only deps (see requirements.txt)
py -3 gui.py                           # opens the queue picker on config.OUTPUT_ROOT
py -3 gui.py --neuron AIAL --chain 0   # open straight onto one chain
py -3 gui.py --reviewer sf             # stamp labels with a reviewer name
```

Or drive it from a notebook / REPL:

```python
from gui import launch
launch(neuron="AIAL", chain_idx=0, reviewer="sf")   # napari.run() blocks until you close it
```

In the window: scrub the chain (the slider is frame index), jump between queued frames (`,` / `.`),
add prompt points (click in the **prompts** layer, pre-loaded with the chain's original seed; `p`/`n`
toggle positive/negative; **reset** restores the saved seed), or paint a correction into the **mask**
Labels layer. Then **re-run image phase** (`R`, re-seed the anchor from your points) and/or **resume
propagation** (`G`, re-track over `PropagationSession` from the current frame) — corrected masks +
`qc.csv` + `state.json` are rewritten on disk, exactly as a fresh batch run would leave them. **Approve**
(`A`) or **reject** (`X`, with an error-type picker) records the chain's disposition in
`output/_review.csv`; per-frame **mark wrong/ok** (`W`/`O`) and the bulk approve/reject log to
`output/_labels.csv` (every queued frame + a uniform sample of un-flagged frames) — the training data
milestone 4.5 will consume.

View conveniences (dock controls + keys): **auto-zoom to the mask** on open and on every jump-to-flagged
so you land on the object, not the whole frame (toggle in the dock, or `Z` / "zoom to mask" to re-fit;
`--no-auto-zoom` to disable); a **point-size** spinbox (default 4 `_sam` px, `--point-size`); and an
opt-in **full-resolution EM background** (`--hires-em` / `launch(hires_em=True)`) — see *Why it looks
low-res* below.

The GPU is lazy: browsing, scrubbing, inspecting flags, painting, and labeling need no predictors
(so a reviewer can work while a background batch holds the card). The SAM2 models are built on the
first re-run/resume only.

This is a **post-batch review tool**, not an inline gate during the first propagation: `batch.py` runs
each chain fully (anchor → propagate → save → QC) headless, then the GUI opens the chains QC *already*
flagged. So when you open a chain, propagation already happened — you are correcting its output and
re-propagating. (Running the GUI *concurrently* with the batch — human clears `flagged` while the GPU
works `pending` — is the §7 *parallel-review* mode, deferred; it needs the file-lock + polling + GPU
arbitration.)

#### Why it looks low-res

Both the mask and the EM frames display at **scale-8** (`_sam` space, ~1152×1154 from a ~9216×9230 EM
frame) because that is the only resolution the pipeline *propagates and saves* at. The default high-res
anchor crop (`crop_anchor`) sharpens only the one-frame *seed*: its box is mapped back to `_sam` and the
crop is discarded — "tier-1 crop sharpens the seed but cannot fix downstream propagation drift"
(`PIPELINE_CONTEXT.md` §4/§7). So:

- The **mask cannot be sharpened in the GUI** — genuinely higher-res masks require propagating at higher
  resolution, the **tier-2 per-chain crop, which is milestone 4.5** (label-gated).
- The **underlying EM image can** be sharpened: `--hires-em` lazily loads the original full-resolution
  tifs as the background and scales the (still scale-8) mask/points to overlay them, so the EM context is
  crisp for judging a correction. It's off by default (full-res frames are ~9216×9230, loaded one at a
  time via dask; falls back to the scale-8 frames if `dask-image` is absent).

#### M4 status — shipped vs. deferred

**Shipped this pass:** queue + review-status ledger (`review_queue`, separate from the batch's
execution manifest); per-frame label store with the anchor-verdict + un-flagged-sample guards
(`labels`); the napari GUI (`gui.py`, walkthrough in [`GUI_GUIDE.md`](GUI_GUIDE.md)) covering
scrub-to-flagged, positive/negative point edits (seed pre-loaded from `state.json`, with a
reset-to-original), mask painting, anchor re-predict, resume-propagation, chain approve/reject with an
error-type picker, and per-frame wrong/ok labeling — all driving the existing `PropagationSession`
+ `run_qc`; plus the view conveniences (auto-zoom-to-mask, tunable point size, opt-in full-res EM
background). Torch-free pieces are unit-tested (`tests/test_labels.py`, `tests/test_review_queue.py`,
24 cases). The napari layer/widget APIs were validated against napari 0.7 (the live Viewer needs an
OpenGL context, so the full window is verified interactively, not in CI).

**Deferred (marked `# [DEFERRED]` in code, with rationale):** genuinely higher-resolution *masks* (the
displayed mask is scale-8 — the tier-2 per-chain propagation crop that would change it is M4.5; the
opt-in `--hires-em` sharpens only the EM background, not the mask); high-res *crop*-space anchor
re-predict (GUI re-predict uses the legacy full-frame `_sam` path that matches the displayed frame); the
*automatic* confidence-gated mask-vs-box video seed (the human-painted-mask seed path is wired; the
auto gate is M4.5 label-gated); a cross-process file lock + live auto-poll + GPU arbitration for
*concurrent* reviewers/batch (single-reviewer is safe; `refresh` is poll-on-demand); and the
build-vs-adopt evaluation of micro_sam's napari plugin (this module is the *build* path). Training a
model on the collected labels is milestone 4.5. See `PIPELINE_CONTEXT.md` §6/§7.

### Run the tests

The torch-free modules have plain-runner + pytest suites, so they can be checked on any box:

```bash
py -3 tests/test_alignment.py        # plain runner, no pytest needed
py -3 tests/test_anchor_select.py
py -3 tests/test_labels.py           # M4 label store
py -3 tests/test_review_queue.py     # M4 queue + review ledger
py -3 -m pytest tests/               # or all at once
```

### Explore in a notebook

For ad-hoc image-mode prompting and visualization, the notebooks plus `UTILS_README.md` are the place to start. Minimal image-mode predict:

```python
import cv2, torch, numpy as np
from sam2_utils import setup, viz, config

predictor, device = setup.build_predictor(size="small", kind="image")
tif = sorted(config.WORM_PATH.glob("*.tif"))[0]
image = cv2.cvtColor(cv2.imread(str(tif)), cv2.COLOR_BGR2RGB)
with torch.inference_mode():
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[4550, 4990]]), point_labels=np.array([1]),
        multimask_output=True)
viz.show_masks(image, masks[np.argsort(scores)[::-1]], scores[np.argsort(scores)[::-1]])
predictor.reset_predictor()
```

---

## Configuration & tuning

Everything you'd tune for a run lives on **`PipelineConfig`** (in `pipeline.py`), in one place:

- **Spaces / resolution:** `scale` (SAM2 input downscale), `save_downscale` (on-disk mask downscale; `== scale` is canonical - no resample, no 2x skeleton offset, and `run_qc` hard-guards this).
- **Anchor crop (default-on):** `crop_anchor`, `crop_size_tif`, `crop_scale` - run image mode on a high-res crop around the node; `crop_anchor=False` falls back to the legacy full-frame path.
- **Per-chain crop / tier-2 (default-off):** `chain_crop`, `chain_crop_pad_tif`, `chain_crop_scale`, `chain_crop_max_px`, `chain_crop_min_tif` - propagate the *whole chain* inside one window sized to its skeleton xy-extent at `chain_crop_scale` (1-2) instead of the scale-8 full frame, for genuinely higher-resolution masks (the space `_pcrop`). Masks are stored at crop resolution and the `CropWindow` is persisted to `state.json` (`ChainState.crop_window`), so QC/`review`/the GUI rebuild the crop space automatically. `chain_crop_scale` is a *target* - a far-wandering chain is read coarser so the SAM2 input stays ≤ `chain_crop_max_px`; `chain_crop_min_tif` (default 1024) **floors** the window so a low-motion chain doesn't over-zoom and lose tracking (an A/B failure mode - see `PIPELINE_CONTEXT.md` §7). Off = the canonical `_sam` full-frame path (and the M1 baseline). Measured A/B (3 AIYL chains): tier-2 cleared the human-review queue on the chains that flagged, at no regression - see `PIPELINE_CONTEXT.md` §7 *Local high-res cropping*.
- **Multimask anchor (default-off):** `multimask_anchor` - ask SAM2 for its 3 candidate anchor masks and auto-select one (node-containment → plausible-area → single-CC → decoder IoU) instead of taking the single-mask output. Near-free (the decoder computes all 3 regardless; `set_image` runs once), touches only the one-frame anchor. Default-off preserves the M1 pixel-for-pixel baseline; reuses the anchor-gate's contain radius + area bounds. (`box-from-radius` is *not* offered - the CATMAID `radius` column is mostly placeholder; mask-vs-box video seeding is deferred to the M4 GUI - see `PIPELINE_CONTEXT.md` §7.)
- **Prompts / seed:** `k_max_neg`, `box_margin`, `seed_negatives` (forward neighbour-node negatives into the video seed; default off).
- **Anchor gate (observational):** `gate_min_area_frac`, `gate_max_area_frac`, `gate_min_largest_cc_frac` - `score_anchor` records a verdict; it doesn't branch yet.
- **QC thresholds:** `qc_area_ratio_bounds`, `qc_temporal_iou_min`, `qc_pred_iou_min`, `qc_skeleton_dilation_px`, and `qc_triage_min_signals` (signals needed to queue a frame for a human; default 2 = intervene).
- **Mask post-processing (default-off):** `postproc_open_px`, `postproc_close_px`, `postproc_keep_largest_cc`, `postproc_fill_holes`.

See `PIPELINE_CONTEXT.md` §5/§7 for what each knob is for and the gotchas behind the defaults.

---

## Notes

- The affine constants in `config.py` were fit at CATMAID z=1293 from 12 landmarks. If you refit on a different section, update `M_AFFINE` / `T_AFFINE` there so every caller picks up the new values.
- Every coordinate transform lives in `alignment.py` (the affine, `tif_to_sam`/`sam_to_tif`, the `catmaid_z`/`file_z` maps, `nm_to_stack_px`, and `CropWindow`). Tag variables with their space suffix (`_tif`, `_sam`, `_crop`, `_cm`) and route conversions through that module rather than writing `/ scale` or `± FILE_Z_OFFSET` inline.
- `setup_device()` enters a bfloat16 autocast context that persists for the session; calling it more than once is safe.
- Masks are stored at `_sam` space (`save_downscale == scale`) as 0/255 uint8 by `pipeline.save_masks`. Don't confuse this with `qc.save_masks` (uint16 instance labels), which the single-object pipeline does not use.
- If a scoring threshold (a `qc_*` or `gate_*` knob) changes mid-campaign, clear or re-score the manifest - rows are written per chain and not rewritten, so a threshold change otherwise silently mixes two configs in `_manifest.csv` (see `PIPELINE_CONTEXT.md` §5).
- Holding both `large` predictors (image + video) resident at once is VRAM-heavy; drop to a smaller `model_size` or build the video predictor lazily if you OOM.
