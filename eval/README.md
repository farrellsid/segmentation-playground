# eval/, Stage 0 evaluation harness

The evaluation harness is the gate everything else waits on: per
[`../roadmap.md`](../docs/explanation/roadmap.md) §5 **Stage 0** and §4.1, we must *fix the ruler
before any further accuracy tuning*. Flag-rate (what every A/B to date leaned on) is being
deprecated as an A/B metric.

## Status

**Stage 0.2 landed: the REAL pipeline now scores against GT** (not the `predict_gt` reimplementation).
Region-overlap + VOI + **ARAND** metrics implemented; ERL metric core implemented. The cross-worm
skeletons are pulled (`data/groundtruth/skeletons_p280/`) and the production `batch.py` runs on
SEM-Dauer 1 via a worm-agnostic frame seam, see **Running the real pipeline on GT** below.

| module | what it does |
|--------|--------------|
| [`groundtruth.py`](./groundtruth.py) | Read the cross-worm VAST GT: parse `VAST_segmentation_metadata.txt` (Nr↔name↔bbox↔hierarchy), and load per-slice binary masks for any segment / neuron. |
| [`metrics.py`](./metrics.py) | `binary_metrics` (IoU/Dice/precision/recall + tp/fp/fn); `variation_of_information` → **VOI_split + VOI_merge** (+ `weighted_voi`); **`adapted_rand`** → ARAND/ARE (SNEMI3D, pair-counting; `are` matches skimage exactly); **`voi_arand`** = unified VOI+ARAND that defaults to **skimage's reference impls** (the CAD/FGNet methodology) with the pure-numpy versions as fallback. |
| [`erl.py`](./erl.py) | **Expected Run Length** (Januszewski 2018): `load_skeletons`, `expected_run_length` / `per_neuron_erl` (Σ run_len²/total, merges→0; split/merge/background breakdown), `sample_node_labels`. Pure core. |
| [`score.py`](./score.py) | `score_region(gt, source, progress=…)` → per-(neuron, slice) and per-neuron metric tables (+ per-neuron `seconds`/`slices_per_s`); `DirPredictionSource` is the reference store. |
| [`gt_dataset.py`](./gt_dataset.py) | **Stage 0.2 adapter**: `GtFrameStore` (per-slice PNG EM → `pipeline.FrameStore`), `build_gt_annotate_df` (x_tif/y_tif baked from the per-section registration), `load_gt_chains` (subset by `neurons`/`neuron_limit`), `build_gt_session_inputs`. |
| [`score_batch.py`](./score_batch.py) | Score a `batch.py` GT run: `BatchPredictionSource` unions a neuron's per-chain `_sam` masks per slice and upscales to the GT grid; CLI emits live progress, `eval_timing.csv`, and `measurement_log.jsonl`. |
| [`scale_registration.py`](./scale_registration.py) | Stage 0.3: derive the full-res `registration.json` by scaling the ¼ fit ×4 (instant, geometrically exact vs a ~1.5 h from-scratch re-fit). |
| [`registration_overlay.py`](./registration_overlay.py) | Stage 0.1 human gut-check: a napari overlay of the full-res VAST EM with two point layers (raw CATMAID coords + registered coords) scrubbable per z, plus click-to-read coordinates. Pure helpers (`build_overlay_table`, `nodes_on_slice`, `nearest_node`) are torch-free and tested. |

Tests: `tests/test_eval_metrics.py` (incl. ARAND vs skimage), `tests/test_eval_groundtruth.py`, `tests/test_eval_erl.py` (`py -3 -m pytest tests/test_eval*.py`; 37 pass).

### Running the real pipeline on GT (Ground Truth SEM-Dauer 1) (Stage 0.2)

`batch.py` is worm-agnostic except the EM source (a `pipeline.FrameStore`) and the skel→image
transform (baked into `annotate_df.x_tif/y_tif`). For SEM-Dauer 1 both come from `eval.gt_dataset`;
the default `TifFrameStore` keeps the target-worm path byte-identical. Run a configurable subset
(the full ~9766-chain run is guarded, you must pass a scope):

```
py -3 batch.py --preset eval --neurons URYVL          # explicit neuron(s)
py -3 batch.py --preset eval --neuron-limit 3         # first N neurons
py -3 batch.py --preset eval --all                    # everything (opt-in; ~9766 chains)
# preset 'eval' (sam2_utils/presets.py) -> SEM-Dauer 1, large model, tier-2 default,
# GT_PRED_DIR/batch_masks + /frames. Override any field: --model-size, --output-root, --no-tier2, --clean.
py -3 -m eval.score_batch --preset eval               # root + out auto-resolved from the preset
# -> region + VOI/ARAND/ERL + eval_frames.csv / eval_neurons.csv / eval_timing.csv / measurement_log.jsonl
```

**Measurement logging.** Every `score_batch` run appends one record to `eval/out_gt/measurement_log.jsonl`
capturing *what* was measured, *against which* GT (dir/downscale/grid), *when* (UTC), *which metrics*, the
per-neuron + overall results, the prediction's provenance (model/scale/crop from `state.json`), and timing.
CSV artifacts: `eval_neurons.csv` (per-neuron region metrics **+ `erl_um`**), `eval_labelmap.csv` (overall
VOI/ARAND/ERL + metric backend), `eval_frames.csv` (per-slice), `eval_timing.csv` (per-neuron seconds/rate).

### Labelmap metrics: VOI / ARAND / ERL ([`score_labelmap.py`](./score_labelmap.py))

> **Which metric is primary?** For the current **sparse, per-neuron** pipeline the appropriate ruler is
> **per-neuron region IoU/precision/recall + ERL** (ERL is skeleton-based, 3D, per-neuron, it fits a
> subset naturally). **VOI/ARAND are secondary here**: they're built for *dense, whole-volume*
> segmentation (CAD/FGNet), so on our scored-neuron subset VOI_merge is **blind to bleed into unscored
> neighbours**, VOI_split/merge largely restate recall/precision, and the values are **not comparable**
> to CAD/FGNet's dense numbers. They become genuinely apt only with a dense labelmap (FUTURE_DIRECTIONS
> §4.3 refinement / R5). Kept because they're cheap and right for that future path.

Region IoU treats each neuron independently. The connectomics metrics need a *labelmap*, all
neurons composited into one per-slice integer map (neuron→id, **first-writer-wins**), built **in
memory at the `_sam` grid** (a full-res uint16 labelmap is ~378 MB/slice). `score_batch` runs these
alongside region (skip with `--no-labelmap`):
- **VOI_split / VOI_merge** and **ARAND**, `eval.metrics.voi_arand`, which defaults to
  **scikit-image's reference implementations** (`variation_of_information` + `adapted_rand_error`,
  `voi = split+merge`), the **CAD/FGNet connectomics methodology**, so the numbers are directly
  comparable. Restricting to GT-foreground == skimage's `ignore_labels=(0,)`. Falls back to this
  module's pure-numpy VOI/ARAND when skimage is absent (verified to match: identical VOI and ARE).
- **per-neuron ERL** + split/merge, skeleton-node sampling through the registration (node coords
  scaled by 1/`save_downscale` to index the `_sam` map), with **neighborhood sampling**
  (`node_sample_radius`, default 2 → dominant non-bg in a 5×5 `_sam` window; the FUTURE_DIRECTIONS §5
  lever, single-pixel sampling of a ~3 px `_sam` neurite is too brittle).
- **Tier-2 aware:** `_pcrop` masks are placed onto the `_sam` frame via each chain's `crop_window`
  (`chain_sam_mask`) before compositing/scoring, both here *and* in the region `BatchPredictionSource`
  (a raw resize of a crop would stretch it across the whole frame; this was a bug, now fixed + tested).

### Stage-0 numbers (June 2026, 3-neuron smoke, PVPR/VA4/AS3; a FLOOR, NOT the final gate)

| run | micro-IoU | VOI (split+merge) | ARAND (ARE) | ERL |
|-----|-----------|-------------------|-------------|-----|
| small model, single-pass `_sam` | 0.022 | 0.875 (0.29+0.59) | 0.162 | ~0% |
| **large model + tier-2 default** | **0.024** | **0.847 (0.22+0.63)** | **0.161** | ~1% (0.22/37 µm) |

Per-neuron (small→large): VA4 **0.012→0.022** (it *kept* tier-2, `_pcrop`, IoU ~doubled, precision up),
AS3 0.095→0.084, PVPR 0.004→0.002 (both *fell back* to `_sam`). Findings:
- **Large model alone barely helps** (slightly hurts the `_sam` neurons), the cross-worm **domain gap
  dominates, not capacity**. **Tier-2 helped where it engaged** (VA4), motivating a lower fallback floor.
- **Tier-2 fell back on 2/3** chains: crop-anchor `image_score` ≈ 0.69, just under the
  `chain_crop_min_image_score = 0.70` floor (a target-worm default; mis-calibrated for cross-worm, lower to ~0.6 to let tier-2 engage). Fallback is **per-chain**; these are single-chain neurons so it
  reads as whole-neuron.
- **Merge/bleed-dominated** (VOI_merge 0.63 ≫ split 0.22; ARAND merge_err 0.26 ≫ split_err 0.03),
  precision ~2.5%, reproduces the `predict_gt` v1 bleed finding.
- ⚠️ **Don't compare VOI/ARAND to FGNet directly.** FGNet (Table 4, AC3/AC4 *dense* mouse cortex) gets
  VOI 0.797 / ARAND 0.069 over a full volume; our VOI 0.847 is over **3 sparse neurons' GT-foreground**
  (far easier, few objects to confuse), so the similar number is *not* comparable quality. Region IoU
  (0.024) and ERL (~1%) are the honest read. *(Caveats: cross-worm = generalization; scale-8 thin
  neurites; no finetuning, all per FUTURE_DIRECTIONS §2/§7.)*

### The GT data, concretely

The VAST export (paths in `sam2_utils.config.GT_*`) is a **16-bit single-channel labelmap** per
z-slice (`*.vsseg_export_s###.png`, PIL mode `I;16`): **pixel value == segment number `Nr`**, `0` ==
Background, not an RGB color image. So the GT mask for segment `Nr` on slice `s` is just
`label_slice(s) == Nr`. As of Stage 0.3 the export is **full-resolution** (`full_scale/`, 9728×9216
== the full-res VAST coords the metadata bboxes are quoted in, so `GT_DOWNSCALE == 1`); **z is 1:1**
(export slice index == metadata z). The `one_fourth_scale/` (2432×2304, 4×) export remains on F: as a
fallback. Per the lab, **every segment present in the metadata is manually confirmed**, so there is no
separate confirmed flag to filter on. 851 slices, 451 segments. *(The full_scale switch required a
full-res `registration.json`, done by scaling the ¼ fit ×4, `eval.scale_registration`; see
Registration below.)*

```python
from eval import GroundTruth, score_region, DirPredictionSource
gt = GroundTruth.from_config()                 # uses config.GT_* paths
gtm = gt.neuron_mask(slice_idx=25, label="RMDVR")   # boolean mask on the GT grid
frames, per_neuron = score_region(gt, DirPredictionSource(pred_root), out_dir="eval/out")
```

## Producing predictions to score

The ruler is in place (region + VOI + ERL); the remaining Stage-0 work is running the *current*
pipeline through it and getting a **trustworthy** number. A first degenerate run with
`eval/predict_gt.py` (small model, points-only seed) produced the first numbers and exposed two things
that reshape how Stage 0 finishes, see [`../roadmap.md`](../docs/explanation/roadmap.md) §5 Stage 0
for the full sub-step plan (0.1-0.4). The short version:

1. **Verify the coordinate transform first (keystone).** The skel→GT registration both *places prompts*
   and *samples node labels*, so a loose transform poisons every number. The dry run showed it: ~50% of
   slices zero-overlap with correctly-sized-but-*displaced* masks, and self-consistency ERL = 0 *with the
   perfect GT as input* (47% of nodes sample off their own segment). The `registration_overlay`
   napari tool (below) overlays the registered nodes on the GT EM so a human can confirm they track
   the right neurites before trusting any score.
2. **Score the real `batch.py`, not a reimplementation.** The honest Stage-0 benchmark is the production
   pipeline (large model, box seed, postprocess, QC) pointed at SEM-Dauer 1 via an argparse dataset
   override, **this supersedes `predict_gt.py` as the scored path.** Its output feeds the scorers below.
3. **Full-res GT (parallel, manual), ✅ EXPORTED + registration re-scaled (Stage 0.3, June 2026).**
   Native-res `full_scale/` (9728×9216, 851 slices) is on F:; `config` now points at it
   (`GT_DOWNSCALE == 1`). This unblocks faithful tier-2 crops and required a full-res registration
   (A ≈ I, not 0.25·I), **done** by scaling the ¼ fit ×4 (`py -3 -m eval.scale_registration`; a
   from-scratch full-res `eval.registration` re-fit is geometrically identical but ~1.5 h of HDD
   decodes, so the ×4 scaling is the cheap exact equivalent). Validated: mean A ≈ I, on-mask 91.7%
   (5-neuron spot check). `registration.json` is now full-res; the ¼ fit is kept as
   `registration_quarter_scale.json`.

`predict_gt.py` is **discontinued** (kept for reference only). It was the points-only scaffold that
validated this harness end-to-end and surfaced the registration problem; the real `batch.py` on GT
(Stage 0.2, above) superseded it as the scored path, so it is no longer maintained and its unfinished
bleed levers retire with it.

**Registration** (`registration.py`): fits skel→GT-mask as a **per-section affine**, a full 2×3
(rotation/scale/shear + translation) per z-slice, fit per slice with one round of outlier rejection,
gaps interpolated + smoothed over z. This replaced the earlier *global linear `A` + per-section
translation* model after `diag_registration.py` (Stage 0.1) showed the residual was structured: a
per-section affine cut the median centroid residual from 19.6 px → **5.1 px** (the realignment carries
per-section rotation/scale a single global `A` can't). The re-fit confirmed it end to end:

| model | median residual | p90 | on-mask (40-neuron) |
|-------|-----------------|-----|---------------------|
| pure ¼-scale (baseline) |, |, | 44.6% |
| global A + per-section translation | 19.6 px | 57.4 px | 67.9% |
| **per-section affine (current)** | **4.7 px** | **17.4 px** | **85.7%** |

Saved to `data/groundtruth/skeletons_p280/registration.json` (the translation model is kept as
`registration_translation_backup.json`). `Registration.transform(xy, z)` and the JSON are unchanged for
consumers, the affine is stored as a per-z `affines` array; old translation-only JSONs still load
(`affines=None` → the legacy path). Re-fit: `py -3 -u -m eval.registration`. Structural/visual check:
`py -3 -u -m eval.diag_registration` (→ residual breakdown + `data/groundtruth/reg_diag/` montage).
**Provenance** (the cross-worm import is genuinely project 280) is corroborated four ways, clean ¼·I
linear part, z-smooth per-section offsets, 97% neuron-name overlap, exact `[0,850]` z-range.

**Known residual (high-priority refinement, see 0.5 in Still open).** The 0.1 human gut-check confirmed
the affine is near-perfect at frame center but its residual grows toward the edges (tens of px at the
rim, consistent with the p90 ~17 px above). An affine maps lines to lines, so it fits tangent to the
true warp at the correspondence centroid and deviates with radius; the likely cause is the elastic
per-section realignment (the EM is `*_realignment_export_*`), which no affine fully matches. Tens of px
is small at full res but misses thin/small cells, and it is the same off-segment-node fraction that caps
the ERL ceiling.

**Advance gate (Stage 0 → Stage 1):** a per-neuron ERL + split/merge breakdown, now qualified as *from
the real `batch.py`, through a verified registration*.

### Verifying the registration (Stage 0.1 gut-check, [`registration_overlay.py`](./registration_overlay.py))

The quantitative checks (on-mask rate, centroid residual, self-consistency ERL) all say the
per-section affine is good; this tool is the human eyeball. It opens a napari viewer of the full-res
VAST EM, scrubbable per z, with two point layers: **raw** CATMAID coords (cyan) and **registered**
coords (magenta). The registration's linear part is ≈ identity at full res (same scale/orientation)
but applies a per-section translation of order ~100-250 px, so the cyan nodes sit that far off the
neurite and the magenta nodes should land on it; the check is that the magenta nodes track the
neurites. Clicking the EM prints the clicked coordinate and the nearest node's name plus its raw
CATMAID (x, y, z), the numbers to paste into the CATMAID project-280 web client to confirm the same
neurite by hand (CATMAID is not queried by the tool).

```
py -3 -m eval.registration_overlay --start-z 400      # --point-size, --no-vnodes, --show-gt-mask
```

## Still open (Stage 0 sub-steps, see FUTURE_DIRECTIONS §5)

- **0.1 Transform ✅ done (per-section affine; on-mask 67.9% → 85.7%, then 91.7% at full res).** The
  `diag_registration` structural check showed the residual was a per-section affine the old model
  missed; `registration.py` now fits that (see the Registration section above), and the
  `registration_overlay` napari tool provides the interactive human gut-check (scrub z, eyeball nodes
  on EM, click-to-read coordinates for a CATMAID cross-check). **Self-consistency
  ERL recovered** (`run_erl --mode self`): node-on-segment 53→**85%**; strict still ≈0 (one stray node
  zeroes a neuron), but **affine + `--merge-tol-frac 0.1` → ERL 11.6 µm = 15% of the 76.3 µm ceiling,
  merges 280→5**, so 0.1 and 0.4 must combine. *Caveat for 0.2:* 15% of nodes still land off-segment
  (residual ~4.7 px vs thin neurites), fragmenting runs, so the ruler's effective ceiling **through the
  current registration + per-pixel sampling is ~11.6 µm, not 76.3**, close it with full-res (0.3) and/or
  neighborhood label sampling in `sample_node_labels`.
- **0.2 Argparse `batch.py` → GT dataset** ✅ done, the real pipeline runs on SEM-Dauer 1 via the
  `FrameStore` seam + `eval.gt_dataset` adapter and is scored by `eval.score_batch` (see *Running the
  real pipeline on GT* above). First small-model smoke landed (micro-IoU 0.022, bleed-dominated).
  **Remaining for the full gate:** a large-model run + composite pred labelmaps so ERL/VOI/ARAND join
  the region metrics.
- **0.3 Full-res GT export** ✅ done (`full_scale/`, 9728×9216, `GT_DOWNSCALE=1`) → faithful tier-2
  crops; **full-res registration** ✅ done via `eval.scale_registration` (¼ fit ×4 → A ≈ I, on-mask
  91.7% spot check; `registration.json` is now full-res, ¼ kept as `registration_quarter_scale.json`).
- **0.4 ERL merge tolerance** so the metric isn't zeroed by a single stray node.
- **0.5 (HIGH PRIORITY) Edge residual in the per-section affine.** The 0.1 gut-check
  (`registration_overlay`) showed the affine is near-perfect at center but off by tens of px at the
  frame edges, enough to miss small/thin cells and likely the same off-segment nodes capping the ERL
  ceiling (~11.6 vs 76.3 µm). Gate the fix on a diagnostic first (verify, don't vibe): extend
  `diag_registration` to plot residual-vs-radius and trial-fit a per-section quadratic / thin-plate
  spline, reporting the *edge* residual and the correspondence density at the rim. That separates a
  too-simple model (higher-order helps) from edge data-starvation (interpolating models extrapolate
  badly past the correspondence hull and can make the edges worse). Only then pick the model.

> Caveat (FUTURE_DIRECTIONS §3, §7): the cross-worm GT measures **generalization**, not
> in-distribution accuracy, treat it as a domain-adaptation benchmark and spot-check on the target
> worm.

See [`../roadmap.md`](../docs/explanation/roadmap.md) §4.1 and §5 Stage 0 for the full reasoning
and sources.
