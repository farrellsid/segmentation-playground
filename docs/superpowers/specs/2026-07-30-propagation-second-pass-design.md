# Propagation second pass: re-segment flagged frames after the chain finishes

Status: design, approved 2026-07-30. Opt-in via a new `batch.py` flag, gated off by default;
nothing changes for existing runs until someone passes it.

## Why

A fresh diagnostic run of `eval.merge_metric` against the real propagation tree
(`target_tier2_s1forced_neg_sam3_merged`, SAM3 backend, 631 chains, 9435 frames) gave:
`foreign_frame_rate=0.252`, `dropout_rate=0.141`, `mean_z2z_iou=0.686`, `mean_centroid_drift_px=3.57`,
`frac_low_iou=0.142`. Two things follow from that, and a comparison already on record:

- Propagation's whole selling point, temporal consistency, is real and now measured: the per-slice
  tree on record has `mean_z2z_iou=0.592` and `mean_centroid_drift_px=5.45`, roughly double the
  low-IoU rate. (Not backend-controlled: that per-slice number is SAM2, this propagation number is
  SAM3, so some of the gap could be backend rather than method.)
- Propagation still bleeds far more than per-slice, and this comparison IS backend-controlled: SAM3
  per-slice's `foreign_frame_rate` is `0.087` (ADR 0017), same backend, so propagation here is
  bleeding almost 3x more on identical model weights.

So propagation trades bleed for consistency, it does not simply lose to per-slice. The goal here is
to cut the bleed and dropout without giving up the consistency, by fixing only the frames that are
actually bad instead of discarding or blindly re-running the whole chain.

The existing blow-up guard (`apply_blowup_guard`, `pipeline/propagate.py`) looks like a precedent
but is not one: it only runs inside `segment_per_slice`, gated on `cfg.per_slice_reseed`, and has
never applied to true video-propagation chains. This design is new ground for propagation, not an
upgrade of an existing lever.

The core idea (student, 2026-07-23, previously unbuilt, parked in project notes): run a second pass
*after* the chain finishes, targeting only the frames QC or the merge metric flagged, rather than
re-anchoring inline mid-propagation. Inline re-anchoring has an open, unverified question, whether
SAM2's or SAM3's video-predictor memory bank actually resets on a mid-chain re-seed, or just layers a
new conditioning frame on top of stale memory. A post-hoc second pass sidesteps that question
entirely: it calls the same stateless `image_predict()` that `segment_per_slice` already uses, and
never touches the video predictor's memory bank at all.

## Architecture

One new function, `apply_second_pass`, in `pipeline/propagate.py` alongside `apply_blowup_guard`.
Runs once per finished chain, opt-in via a new `PipelineConfig.second_pass: bool = False` field
(mirrors `blowup_guard`'s own pattern), and only for propagation-mode chains
(`not cfg.per_slice_reseed`; per-slice already has its own guard, so per-slice chains are untouched
here). No new CATMAID read is needed: `Session.annotate_df` is already built once at `batch.py`
startup with `x_tif`/`y_tif` attached, the exact same shape `eval.merge_metric.load_node_table()`
produces, so `apply_second_pass` is simply handed the already-loaded table.

Three sub-steps: trigger, re-segment, fallback.

### 1. Trigger: chain-level pre-filter, then a frame-level gate

Called from `_run_one_chain` in `batch.py`, right after the chain's final `state` comes back from
`_run_chain_once` (post-QC, post any tier-2 rerun) and before `save_state`. Skip the whole chain
immediately if its `qc.csv` has no flagged rows (the same `queue`/`intervene`/`flag` columns
`build_triage_queue` already reads), so a clean chain costs nothing extra. Otherwise call
`eval.merge_metric.score_chain(chain_dir, neuron, nodes_by_z, radius)` once for the chain. This
reuses `pipeline.chain_masks_in_sam` internally, which already resolves each frame's own crop offset,
so `_sam` and tier-2 `_pcrop` chains are both handled without extra code. A frame is marked for the
second pass if its `score_chain` record has `empty=True` (dropout), `n_foreign > 0` (bleed), or
`own_contained=False` (lost its own node).

### 2. Re-segmentation: neighbour mask-prompt

For each flagged frame z, search outward by `|z - z'|` among the same chain's other frames for the
nearest z' that `score_chain` did not flag. Load that neighbour's saved mask; a tier-2 chain has one
fixed `crop_window` for its whole run (confirmed: `state.json` stores it once per chain, not per
frame), so the neighbour's mask is already in the flagged frame's crop-space, no remapping needed.

Two pieces need building, verified directly against the installed SAM2 predictor rather than assumed:

- `image_predict` (`pipeline/predict.py`) has no `mask_input` parameter today; it forwards only
  `point_coords`/`point_labels`/`box`/`multimask_output` to `image_predictor.predict()`. Add an
  optional `mask_input: np.ndarray | None = None` parameter and forward it unchanged.
- `SAM2ImagePredictor.predict`'s real docstring says `mask_input` must be a low-resolution `1x256x256`
  logits array, "typically coming from a previous prediction iteration," not a full-resolution
  boolean mask. A saved neighbour mask is full-resolution and boolean, so it needs converting: a new
  helper, `mask_to_low_res_logits(mask: np.ndarray) -> np.ndarray`, resizes the mask to 256x256 and
  maps it to a logit-like array (large positive where True, large negative where False), so
  thresholding at 0 inside SAM2 recovers the same shape.

Call `image_predict(image_predictor, frame_image, prompts, mask_input=mask_to_low_res_logits(neighbour_mask))`.
`prompts` cannot reuse the chain's original anchor seed points here: only the anchor frame has
explicit seed points in a propagation chain, every other frame's mask comes from the video
predictor's tracking, not a per-frame prompt. Instead, build a single positive point from that
frame's own skeleton node position, `nodes_by_z[z]` filtered to this neuron (the same table the
trigger step already used, and confirmed directly against the real tree: every sampled neuron has a
node at every z within its span, so this is always available for a flagged frame inside the chain's
range). The mask hint does the heavy lifting; the point exists mainly to disambiguate which object
the mask hint refers to.

### 3. Fallback: reuse the existing guard, do not invent a new one

After re-segmenting, one cheap check: the new mask is non-empty and contains the frame's own node
(the same `own_contained` check `score_chain` already computes). If that check fails, or no
unflagged neighbour exists anywhere in the chain, fall back to exactly `apply_blowup_guard`'s
existing behaviour: copy the nearest accepted (non-flagged) frame's mask, and zero that frame's
`frame_conf`/`pred_iou` so QC queues it for a human. This is a floor against making a frame worse,
not a second scoring pass; there is no attempt to compare the re-predicted mask against the original
propagated mask and pick a winner (that fuller approach was considered and set aside for a first
landing, see Out of scope below).

## Disk writes

Any frame the second pass touches gets its saved mask PNG rewritten in place (same filename and z, the
same way a GUI correction rewrites masks today) and its `qc.csv` row gains a new `second_pass` column:
`"corrected"` (re-segmentation passed the sanity check), `"guard_fallback"` (fell back to
neighbour-copy), or blank for untouched frames. `_triage.csv` already rebuilds from `qc.csv` on every
run, so it reflects the corrected state afterward, not the stale pre-pass flags.

## CLI / config

- `PipelineConfig.second_pass: bool = False` (`pipeline/config.py`, alongside `blowup_guard`).
- `batch.py --second-pass` flag, off by default, threaded the same way `--postprocess` already
  overrides its own config field.
- No new preset needed. Existing propagation presets (e.g. `original_tier2_s1forced_neg`) can opt in
  through the flag alone, since this is a pure post-hoc addition to a chain that already used
  propagation.

## Testing

- Unit tests for the trigger logic against synthetic `score_chain`-shaped records (empty, foreign,
  and own-not-contained combinations), in the style of `test_blowup_guard.py`: build small synthetic
  chains and assert exactly the right frame indices get selected.
- Unit tests for `mask_to_low_res_logits`: a synthetic boolean mask round-trips through it and a
  threshold-at-0 check back to the same shape, allowing for the 256x256 resize.
- Unit tests for the neighbour search and fallback: a synthetic multi-frame chain where the nearest
  neighbour is unambiguous, one where no unflagged neighbour exists (fallback path), and one where
  the re-predicted mask fails the sanity check (fallback path).
- A real smoke test before trusting this on a full re-run: apply `apply_second_pass` to one real
  flagged chain from the existing `target_tier2_s1forced_neg_sam3_merged` tree (F: drive) by hand,
  and report the before/after `own_contained`/`n_foreign`/`empty` values for the frames it touches.

## Out of scope for this landing

- Per-slice chains (`cfg.per_slice_reseed=True`): already covered by `apply_blowup_guard`, untouched
  here.
- Generating multiple candidate re-segmentations and scoring them to pick a winner: considered during
  design and set aside in favour of the simpler single neighbour-mask-prompt mechanism. Worth
  revisiting if the simple version's real numbers disappoint.
- True inline mid-propagation re-anchoring, touching the video predictor's memory bank directly:
  still blocked on the unverified SAM2/SAM3 memory-reset question. This design's whole point is to
  avoid needing that answer.
- A standalone post-hoc driver that could apply this to an already-completed tree without a re-run:
  explicitly deferred. This lands as a `batch.py`-only flag for future runs, not a tool for the
  22-neuron tree already on disk.
