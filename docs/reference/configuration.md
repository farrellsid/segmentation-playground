# Configuration

There are two configuration surfaces. Static, per-box facts live in `sam2_utils/config.py`. Per-run
tunables live on `PipelineConfig` in `pipeline.py`. Default run setups are bundled in named presets
in `sam2_utils/presets.py`.

## config.py (static constants)

Paths and constants that change per machine or per dataset, not per run:

- `WORM_PATH`: the raw EM `.tif` stack. Overridable with the `SAM2_WORM_PATH` env var (used on the
  cluster, where the stack lives under project storage); unset, it uses the local default. See
  [../how-to/run-on-narval.md](../how-to/run-on-narval.md).
- `DATA_DIR`, `CSV_PATH`, `CHAINS_PATH`, `ROOTS_PATH`: the CATMAID-derived inputs (resolved from the
  repo location).
- `OUTPUT_ROOT`, `FRAMES_ROOT`: the mask-output and JPEG-scratch roots.
- The SAM2 checkpoint registry (tiny, small, base_plus, large).
- CATMAID URL and project id, `STACK_RESOLUTION_NM`, `FILE_Z_OFFSET`.
- The fitted affine `M_AFFINE` and `T_AFFINE` (see [coordinate-spaces.md](coordinate-spaces.md)).
- The cross-worm GT paths (`GT_*`) for evaluation.

## PipelineConfig (per-run knobs)

Everything you would tune for a run lives here, in one place. Defaults reproduce the original
single-chain run.

- Spaces and resolution: `scale` (SAM2 input downscale), `save_downscale` (on-disk mask downscale;
  equal to `scale` is canonical, see [ADR 0006](../adr/0006-canonical-mask-space.md)), `image_size`
  (override SAM2's internal input resolution, default 1024; None keeps the checkpoint default). SAM2
  resizes every frame to `image_size`, so raising it is the only in-model way to feed more pixels
  than scale/cropping give; verified post-build in `setup.py`, memory grows roughly quadratically.
- Anchor crop (default on): `crop_anchor`, `crop_size_tif`, `crop_scale`. Runs image mode on a
  high-res crop around the node. Off falls back to the full-frame path.
- Per-chain crop, tier-2 (default off, auto-on for flagged chains): `chain_crop`,
  `chain_crop_pad_tif`, `chain_crop_scale`, `chain_crop_max_px`, `chain_crop_min_tif`,
  `chain_crop_collapse_size_tif`, `chain_crop_fallback`, `chain_crop_min_image_score`,
  `chain_crop_from_mask`. See [ADR 0009](../adr/0009-tier2-crop-fallback.md).
  `chain_crop_pad_tif` defaults to 512 `_tif` px: windows sized from the first-pass mask
  (or the skeleton) that looked fine often clip the cell, so every tier-2 window gets a
  generous margin (`crop_scale` bumps coarser if the wider extent exceeds
  `chain_crop_max_px`). `chain_crop_collapse_size_tif` (default 1024 `_tif` px, 0 to
  disable) is the collapse fallback for `chain_crop_from_mask`: when the first pass left
  masks but they collapsed to no usable foreground, the window becomes a fixed square of
  this size centred on the anchor node instead of a skeleton-only guess.
- Multimask anchor (default off): `multimask_anchor` asks SAM2 for its three candidate anchor masks and
  auto-selects one by `(contains the positive node, plausible area, single connected component, SAM
  IoU)`. `multimask_exclude_neg` (default off, only consulted when `multimask_anchor` is on) adds an
  anti-bleed term: among candidates that contain the positive node, prefer one that contains none of
  the negative neighbour nodes. `multimask_generous` (default off, also consulted only when
  `multimask_anchor` is on) instead changes the tie-break to prefer the LARGER gate-passing candidate,
  so a soma mask includes the nucleus and reaches the outer membrane; it stays capped by the area gate,
  so a whole-frame blob does not win while any gate-passer exists. The `eval` preset turns
  `multimask_anchor` on. See [ADR 0012](../adr/0012-node-anchored-multimask-selection.md).
- Per-slice re-seeding (default off): `per_slice_reseed` replaces whole-chain video propagation with
  independent per-slice image-mode segmentation, each slice re-seeded from its own skeleton node inside
  the chain crop, so SAM2 memory cannot carry the wrong cell across slices. When off, the propagation
  path is byte-identical. See the Phase 1 spec under `docs/superpowers/specs/`.
- Prompts and seed: `k_max_neg`, `box_margin`, `box_margin_frac`, `seed_negatives`, plus the seed
  shape knobs. See [ADR 0008](../adr/0008-video-seed-box-vs-mask.md).
- Anchor gate (observational): `gate_min_area_frac`, `gate_max_area_frac`, `gate_min_largest_cc_frac`.
  Records a verdict; does not branch yet.
- QC thresholds: `qc_area_ratio_bounds`, `qc_temporal_iou_min`, `qc_pred_iou_min`,
  `qc_skeleton_dilation_px`, `qc_triage_min_signals`. See [qc-signals.md](qc-signals.md).
- Mask post-processing (master toggle `postprocess_masks`, default off): the morphological
  baseline `postproc_open_px`, `postproc_close_px`, `postproc_keep_largest_cc`,
  `postproc_fill_holes`, plus the size-aware ops `postproc_remove_islands_min_size`
  (keep all components above a size floor, not just the largest),
  `postproc_fill_small_holes_area` (fill only small holes, keep large cavities), and
  `postproc_smooth_radius` (disk close-then-open to smooth frayed edges). The size-aware
  ops run only when their value is > 0, so they leave the baseline unchanged by default.
  `batch.py --postprocess` / `--no-postprocess` flips the master toggle for an A/B run.

## Presets

A preset (`sam2_utils/presets.py`) bundles the worm, paths, model, tier-2 settings, and default
neurons, so a run is `--preset <name> [--neurons ...]` instead of a long flag string. Two ship today:
`original` (the target worm) and `eval` (the cross-worm GT). Any CLI flag overrides the preset.

## Membrane metric parameters (`sam2_utils/membrane.py`)

The v1 membrane map and its three detector primitives (`spanning_membrane`, `boundary_on_membrane`,
`underfill_fraction`) take a small set of tunables, exposed as module constants and forwarded through
`eval.merge_metric`'s membrane pass:

- `sigmas` (default `(1, 2, 3)`): the Sato ridge-filter scales `membrane_map` runs the dark-ridge
  response at, before normalising by the response's own 99th percentile.
- `tau` (default `0.5`): the threshold on the normalised `[0, 1]` membrane map above which a pixel
  counts as membrane. Shared by all three detectors; also a `merge_metric` CLI flag.
- `f` (default `0.15`): the minimum area of a component, as a fraction of the mask's own area, for
  `spanning_membrane` to count it as a border candidate. Filters out membrane-adjacent speckle.
- `tol` (default `2` px): the dilation radius `boundary_on_membrane` allows between the mask
  perimeter and a membrane pixel before counting that perimeter pixel as off-membrane. Also a
  `merge_metric` CLI flag.
- `k` (default `6` px): the bound on `underfill_fraction`'s outward flood, so a broken ridge leaks
  only locally rather than into an unrelated neighbour.

All five are resolution-aware for the `_sam` grid the metric runs on (the coarse, scale-8-ish working
resolution most runs save at), and comparative rather than absolute: they are tuned for grading one
run against another at a fixed scale, not for asserting a boundary is correct in isolation. See
[ADR 0016](../adr/0016-membrane-map-border-to-border-bleed-detection.md) for why the spanning
criterion works border-to-border, and [cli.md](cli.md) for the `merge_metric` flags.
