# Phase 1 close-out: per-slice blow-up guard + generous-first / negatives-in-crop bundle

Status: design, approved 2026-07-17. Two independent behaviour changes that finish Phase 1, plus the
presets and the CCDB batch to measure them. Both are gated off by default; nothing changes for
existing runs.

## Why

The retro-eval settled two things. Per-slice re-seeding has the cleanest bulk of any method (dropout
0.002, lowest mild-bleed) and is the cheapest (~7 s/chain), but a handful of slices blow up to the
whole worm cross-section (`total_foreign` 17,481, from 2-frame whole-worm spikes seen in
`docs/figures/per-slice-blowup/`). And negatives cut bleed 24 to 44% in the crop, a large effect the
old flags missed. This round adds the source-side guard that tames per-slice's tail, and a two-pass
config that keeps the first pass generous (so the crop is not clipped) while putting negatives where
they pay off (the crop).

## Feature A: per-slice blow-up guard

A post-pass at the end of `segment_per_slice` (`pipeline/propagate.py`), gated by `cfg.blowup_guard`
(default False). Per-slice segmentation is order-independent, so the whole chain's masks are already
collected in `video_segments` before return, which is where the guard runs.

- Compute the median area over the non-empty masks in the chain.
- A mask is a blow-up if its area exceeds `cfg.blowup_area_factor` times that median (default 25.0,
  generous: the observed real blow-ups were ~2000x the median and normal size variation is under 10x).
- Replace each blown-up mask with the nearest accepted slice's mask by frame index (`argmin |i - j|`
  over accepted j). If no accepted neighbour exists, leave it empty.
- Mark guarded frames: set their `frame_conf` and `pred_iou` to 0.0 so the existing QC confidence
  flag queues them for human review (a substituted neighbour mask is a stand-in, not a real
  segmentation of that slice), and log the guarded count.

Config (in `sam2_utils/config.py`, `PipelineConfig`):
- `blowup_guard: bool = False`
- `blowup_area_factor: float = 25.0`

Skip the guard (no-op) when there are fewer than a few accepted masks or the median is 0, so a short
or mostly-empty chain cannot establish a spurious baseline. The guard never runs on the video-propagate
path (it is a `segment_per_slice`-only post-pass), so non-per-slice runs are byte-identical.

## Feature B: generous-first-pass, negatives-in-crop bundle

Reuses two existing mechanisms: `tier2_all` (run both passes on every chain, not just flagged ones)
and `chain_crop_from_mask` (size the tier-2 crop from the first pass's `_sam` mask bbox, unioned with
the skeleton box). The only new capability is per-pass seed config: today the tier-2 rerun in
`batch._run_one_chain` does `replace(cfg, chain_crop=True)`, so the second pass inherits the first
pass's seeds. We add optional tier-2 seed overrides so the two passes can differ.

Config (in `PipelineConfig`), each `None` meaning "inherit the base value":
- `tier2_k_max_neg: int | None = None`
- `tier2_seed_negatives: bool | None = None`
- `tier2_multimask_generous: bool | None = None`

`batch._run_one_chain`'s rerun becomes:

```
overrides = {}
if cfg.tier2_k_max_neg is not None:        overrides["k_max_neg"] = cfg.tier2_k_max_neg
if cfg.tier2_seed_negatives is not None:   overrides["seed_negatives"] = cfg.tier2_seed_negatives
if cfg.tier2_multimask_generous is not None: overrides["multimask_generous"] = cfg.tier2_multimask_generous
state = _run_chain_once(session, replace(cfg, chain_crop=True, **overrides), ...)
```

Nothing else in the rerun changes. When all three overrides are None (every existing preset), the
rerun is `replace(cfg, chain_crop=True)` exactly as today, so current behaviour is unchanged.

Data flow for the bundle: pass 1 runs `_sam` from the base cfg (generous on, negatives off) and writes
`_sam` masks + `state.json` (crop_window null) + `qc.csv`. Pass 2's `chain_crop_from_mask` reads those
`_sam` masks (declining only if the prior masks were `_pcrop`, which they are not here), unions their
bbox with the skeleton box to size the crop, and re-segments in the crop with the tier-2 seed overrides
(negatives on). This all happens inside one `_run_one_chain` call, sequentially, so pass 2 sees pass 1's
freshly-written masks.

## Presets (`sam2_utils/presets.py`)

- `original_perslice_only_guard`: `original_perslice_only` + `blowup_guard=True`.
- `original_perslice_guard`: `original_perslice` + `blowup_guard=True`.
  (A/B against the guard-off originals already on disk.)
- `original_genfirst_negcrop`: `tier2_all=True`, `chain_crop_from_mask=True`; base seeds
  `multimask_generous=True, k_max_neg=0, seed_negatives=False`; tier-2 overrides
  `tier2_multimask_generous=False, tier2_k_max_neg=3, tier2_seed_negatives=True`. Inherits the
  EXP_NEURONS subset and `chain_crop_scale` from the `original` family.

The tier-2 crop pass is deliberately not generous: the first pass is generous so the crop is not
clipped, the crop pass wants precision (negatives, tight mask), so generosity is off there.

## CCDB batch

All prepped to submit together (scripts in `cluster/`, design in the run-on-narval how-to):

1. Three GPU arrays via `run_exp.sh`, one per new preset: `original_perslice_only_guard`,
   `original_perslice_guard`, `original_genfirst_negcrop`.
2. A `run_merge_exp.sh` merge per preset, each an `afterok` dependency of its array.
3. A CPU big-memory job: `retro_eval --membrane --min-scale 1` over all existing trees, to finish the
   report (whole-image merge-metric + membrane for every method). No GPU.
4. A follow-on `afterok` job that runs `retro_eval` over the three new merged trees so the comparison
   CSV is produced on-cluster; pull it back with `scp`.

The exact command list is delivered with the implementation, not fixed in this spec (paths depend on
the run).

## Testing (CPU, torch-free)

- Guard, synthetic `video_segments` (no torch): one mask at 2000x median is replaced by its nearest
  accepted neighbour and its frame is flagged (frame_conf 0); a mask at 10x median is untouched; an
  all-empty or too-short chain is a no-op. Test the guard as a small pure function over the
  `(video_segments, frame_conf, pred_iou)` dicts so it needs no image predictor.
- Per-pass override, unit-test that the rerun config carries the tier-2 seed values while the base cfg
  is unchanged: build a cfg with the three `tier2_*` overrides set, call the override-assembly logic,
  assert the `replace`d cfg has the tier-2 values and a cfg with all-None overrides reproduces
  `replace(cfg, chain_crop=True)` exactly. No GPU.

## Risks and limits

- The guard's neighbour substitution is a stand-in, not a true segmentation; hence the forced flag so a
  human confirms it. It assumes adjacent slices are similar, which holds for the isolated 1-2 frame
  spikes we saw but would degrade if blow-ups clustered (the log surfaces that case).
- The bundle depends on `chain_crop_from_mask`, whose clipping benefit was measured only on flags
  before; the merge-metric A/B here is its first principled test. If the generous first pass over-grows
  the crop, that shows up as slower compute and possibly more bleed, which the metric will catch.
- `blowup_area_factor=25` is chosen from one chain's montage; it is a starting default, tune from the
  guarded-frame counts the first run logs.
