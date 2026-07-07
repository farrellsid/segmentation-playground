# Full-resolution segmentation experiments (design)

Date: 2026-07-06
Status: variants 1-3 implemented and tested; variant 4 (stitch) designed, not yet built.

A head-to-head of four ways to spend the Narval cluster's compute on segmentation
resolution for the target worm, so we can download the outputs and compare which raises
mask quality. Runs on a fixed neuron subset, one output tree per variant, reviewed
visually (the target worm has no ground truth, so no automatic scoring).

## Why (the constraint that shapes all of this)

SAM2's image encoder resizes every frame or crop to a fixed internal `image_size` (1024
by default), verified in the source. So feeding "the whole image at full resolution" does
not raise the effective resolution: a roughly 9.7k-pixel frame is downsampled about 9.5x
to 1024, versus about 1.2x at the current scale-8. The only ways to give SAM2 more real
pixels are to crop (so 1024 covers a smaller area) or to raise `image_size` itself. The
four variants below span those levers.

## The four variants

All share the model (`large`), the seed knobs, and the neuron subset
(`presets.EXP_NEURONS` = `KEY_NEURONS` + `AVAL`, 16 neurons), so only the resolution
strategy differs. Each writes a distinct output tree, none touching the previous full
run's `target_shards` / `target_merged`.

| # | Preset | Strategy | Type | Expectation |
|---|--------|----------|------|-------------|
| 1 | `original_fullres` | whole frame at `scale=1`, no tier-2 | config only | equal-or-worse than scale-8 (SAM2 still sees 1024); a measured baseline |
| 1c | `original_wholeimg_s4` | whole frame at `scale=4`, no tier-2 (control) | config only | similar to variant 1: same 1024 view, only the pre-downsample path differs |
| 2 | `original_tier2forced` | tier-2 crop on every chain, `chain_crop_min_image_score=0`, no fallback | config only | the high-res mechanism that already works, applied universally |
| 3 | `original_bigimg` | `image_size=2048`, `scale=4` frames (~2432 px), no tier-2 | code (config knob + `setup.py` hydra override) | real extra pixels, but off-distribution and may OOM on a 40GB A100 |
| 4 | `original_stitch` | per-chain full-res crop tiled into <=1024 windows, single-object masks unioned | new feature (crop + orchestrator) | not yet built; see below |

Variant 1c is a control that empirically checks the "why" above: it is variant 1 with only
`scale` changed (1 -> 4) and `image_size` left at the default 1024. If its masks come out
similar to variant 1's, the 1024 bottleneck is confirmed and whole-image `scale` does not
change effective resolution; a large difference would refute it. It pairs with variant 3,
which instead changes `image_size`, so the two isolate the scale knob from the input-size
knob. Expect small deltas, not identical masks: the two scales reach 1024 by different
pre-downsample filters and JPEG sizes.

`save_downscale` tracks `scale` in every variant (the qc node-lookup guard requires
`scale == save_downscale`).

## Implementation (variants 1-3)

- `PipelineConfig.image_size: Optional[int] = None`. When set, `setup.build_predictor`
  (both image and video) passes `++model.image_size=N` as a hydra override.
- Because a wrong override key would silently no-op (a force-added key nothing reads),
  `setup._assert_image_size` reads the built model's actual `image_size` and raises if it
  does not match the request. We cannot watch the cluster run, so a silent no-op must
  become a hard failure the shard log shows. This is the safety net for variant 3.
- Three presets in `sam2_utils/presets.py` (see the table). `EXP_NEURONS` is baked into
  the presets so every variant runs the identical subset without relying on the submit
  command.

## Logging (post-hoc, since the run is unobserved)

`batch.write_run_meta` writes `_run_meta.json` at each run's output root: the preset, the
resolved resolution and tier-2 knobs, the git commit and dirty flag, host, argv, the
neuron scope, the full `PipelineConfig`, and the video predictor's actual `image_size`
after build. `cluster/merge_shards.py` copies a shard's `_run_meta.json` up into the
merged tree so the downloaded result carries its own provenance. Combined with the
per-chain `state.json` (phase timings, `fell_back_to_sam`, `crop_image_score`,
`qc_summary`) and the `_manifest.csv` / `_timing.csv` already written per run, that is
enough to reconstruct what each chain did.

## Cluster execution

Chunked as a Slurm array, 2 neurons per task (8 chunks, `cluster/exp_neuron_chunks.txt`),
so the variants run in parallel, AVAL's many chains are isolated to one task, and a single
task failing does not lose the variant. `cluster/run_exp.sh` is submitted once per variant
via `--export=ALL,EXP_PRESET=...`; shards land in `/scratch/$USER/<preset>_shards/chunk_i`.
`cluster/run_merge_exp.sh` (afterok dependency, same `EXP_PRESET`) merges each variant into
`/scratch/$USER/<preset>_merged`. The full submit sequence is in the header of
`cluster/run_exp.sh` and in `docs/how-to/run-on-narval.md`.

The fourth comparison point is the existing full target-worm run (`original` preset,
tier-2 with the default fallback floor), already on `/scratch`.

## Variant 4: stitch (designed, not yet built)

Reading `crop.py` confirmed there is no config shortcut: since SAM2 resizes any crop to
1024, a full-res crop only beats tier-2 if it is actually tiled into <=1024 windows and
the results combined. Because each chain is a single object, combining is a union in crop
space (no cross-object arbitration needed), which is tractable but is new orchestrator
logic, not a flag:

1. Build the chain's crop at `crop_scale=1` (full res) instead of bumping it coarser when
   it exceeds `chain_crop_max_px`.
2. If the crop's longest edge exceeds the input cap, tile it into overlapping windows each
   <= the cap, seeding each tile from the chain nodes that fall in it.
3. Run image-predict + video-propagate per tile; remap each tile's mask to crop space
   (reusing `remap_mask_to_window`) and union.

This needs a CPU test for the tiling and union geometry plus a local smoke before it goes
on the cluster, so it is built as the immediate next step rather than rushed onto tonight's
submit. Because the shard folders are independent, variant 4 can be submitted separately
when ready without redoing variants 1-3.

## Caveats

- No verified source evaluates SAM2 on thin C. elegans neurites, and variant 3's numbers
  are off-distribution, so treat all of this as measurement, not expected wins.
- Variant 3 may OOM on a 40GB A100 (memory is roughly quadratic in `image_size` times
  about 300 stored frames). An empty `bigimg` shard with an OOM in its log is
  expected-failure, not a bug; the `_run_meta.json` records what was attempted.
- Variants 1 and 3 write masks at full or quarter resolution (`save_downscale` 1 or 4), so
  their output trees are much larger on disk than the scale-8 runs.
