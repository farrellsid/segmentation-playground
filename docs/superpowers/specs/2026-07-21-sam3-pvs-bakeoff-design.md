# SAM3 PVS tracker bake-off, design

Date: 2026-07-21
Status: proposed (awaiting review)
Related: `docs/explanation/roadmap.md` (Phase 1 per-slice, Phase 4 paradigm gate),
`api_docs/SAM3/` (Meta reference notebooks + checkpoint README, local reference)

## Context

A postdoc shared a SAM 3 checkpoint (`F:\sam3\huggingface`, HuggingFace format) and its
output looks better than our current SAM 2 masks. SAM 3 bundles two capabilities: Promptable
Concept Segmentation (text and exemplar prompts, all instances at once) and Promptable Visual
Segmentation (points, boxes, masks, one instance per prompt). PVS is the direct analogue of our
CATMAID point and box pipeline. Meta ships PVS as `Sam3Tracker` and `Sam3TrackerVideo`, described
as a same-API drop-in replacement for SAM 2 and SAM 2 Video.

We want a measured answer to one question before committing to any pipeline change: on our data,
is SAM 3 better than SAM 2, for both segmentation strategies we run today.

## Goal

Compare four method cells on the target worm and score them on the same GT-free ruler:

| | propagation | per-slice |
|---|---|---|
| SAM 2 | `pipeline.propagate` (existing) | `pipeline.segment_per_slice` (existing) |
| SAM 3 | via a SAM 3 video adapter | via a SAM 3 image adapter |

The comparison must be apples-to-apples: identical chains, anchor seeds, crop, multimask
selection, and scoring. The only variable is the model.

## Non-goals

Not in this work: any PCS or text-prompt path, any finetuning, wiring SAM 3 into `batch.py`,
manifest or resume integration, a CCDB job, or cross-worm GT scoring. SAM 2 stays the pipeline
default throughout, so the AVAL pixel-for-pixel reproduction is untouched.

## Environment (verified 2026-07-21)

- `transformers` 5.13.1 already exposes the tracker classes (`Sam3TrackerModel`,
  `Sam3TrackerProcessor`, `Sam3TrackerVideoModel`, `Sam3TrackerVideoProcessor`,
  `Sam3TrackerVideoInferenceSession`), so no new install is needed and the working SAM 2
  environment is not disturbed.
- `torch` 2.12.0 + CUDA 13.0, local GPU is an RTX 3050 6GB. bfloat16 plus CPU offload keep VRAM
  bounded; a long chain may still be Narval-only, which the harness reports rather than assumes.
- The checkpoint loads from the local path `F:\sam3\huggingface` via `from_pretrained`. F: is
  known to be flaky under load; the harness fails loudly on a read error rather than producing a
  partial run.

## Architecture

Two phases. This spec covers Phase 1 only.

### Phase 1: adapters plus a bake-off harness

The fair way to run the SAM 3 cells is to feed SAM 3 through the existing `segment_per_slice`
and `propagate` functions, so every step other than the model is shared with SAM 2. That requires
two adapters that present the predictor interface those functions already call.

**`Sam3ImagePredictor` adapter.** Mimics the subset of `SAM2ImagePredictor` that
`pipeline.predict.image_predict` uses:
- `set_image(image_rgb: np.ndarray) -> None`
- `predict(point_coords, point_labels, box, multimask_output) -> (masks, scores, logits)`

Internally it wraps `Sam3TrackerModel` plus `Sam3TrackerProcessor`. Point and box coordinates are
pixel coordinates in both APIs (confirmed from the HF tracker examples), so no rescaling is needed
at this boundary. `masks` are returned at the input image resolution via
`processor.post_process_masks(..., original_sizes=[(H, W)])`; `scores` map from the output
`iou_scores`; `logits` map from the low-resolution `pred_masks` so that `image_predict`'s
mean-foreground-sigmoid proxy and any mask-input chaining keep working. With this adapter,
`segment_per_slice` and its multimask selection, area gate, and blow-up guard run unchanged.

**`Sam3VideoPredictor` adapter.** Mimics the surface `pipeline.propagate.PropagationSession` calls:
- `init_state(video_path, offload_video_to_cpu=True, ...) -> inference_state`
- `reset_state(inference_state) -> None`
- `add_new_points_or_box(inference_state, frame_idx, obj_id, box, points, labels, clear_old_points) -> ...`
- `add_new_mask(inference_state, frame_idx, obj_id, mask) -> ...`
- `propagate_in_video(inference_state, reverse, start_frame_idx, max_frame_num_to_track) -> iterator of (frame_idx, obj_ids, mask_logits)`

Internally it wraps `Sam3TrackerVideoModel` plus `Sam3TrackerVideoProcessor` and a
`Sam3TrackerVideoInferenceSession`, translating to HF's `init_video_session`,
`add_inputs_to_inference_session`, `propagate_in_video_iterator`, `reset_inference_session`, and
`post_process_masks`. `init_state` loads the on-disk JPEG frames (the same
`{frame_idx:05d}.jpg` layout `prepare_video_frames` writes) into the session. Masks are returned at
frame resolution and thresholded exactly as `PropagationSession._collect` expects. With this
adapter, `propagate` and `run_bidirectional` run unchanged.

The adapters are the core of the eventual production backend, so this is deferred CLI wiring, not
throwaway work. They live in a new module (proposed `sam2_utils/sam3_backend.py`) that the library
may import but that itself imports torch and transformers lazily, so the import-direction test and
the CPU-only default are preserved.

**Bake-off harness** (proposed `experiments/sam3_bakeoff.py`, a driver, not library code). It:
1. Loads a chain with `review.load_chain` plus `pipeline.load_state` (the `coprop_lab` pattern),
   default chains AIAL chain_05 (short, ~17 frames, the fail-fast case) and AIAL chain_00
   (long, ~113 frames, anchor 56, a realistic mid-stack propagation). The chain list is a CLI
   argument with these as defaults. Root is `config.OUTPUT_ROOT` (the sensory-ablated target worm).
2. Builds the anchor prompts once per chain (`predict.build_prompts`) and runs all four cells at
   the canonical `scale = 8` grid with one shared config, so seeds and crop match.
3. Scores each cell with the target-worm merge-metric (foreign-node containment and dropout,
   `eval.merge_metric`) plus the membrane detectors (underfill, mild-bleed, `sam2_utils.membrane`).
4. Writes per-cell overlay PNGs under `docs/figures/sam3-bakeoff/` and a four-row summary table
   (foreign-node rate, dropout, underfill, mild-bleed, wall-clock, peak VRAM) to a results log.

### Phase 2 (sketch, out of scope here)

If SAM 3 wins, promote the adapters behind a `Backend` selector and add `--backend {sam2,sam3}`
to `batch.py`, with manifest, resume, and CCDB support, SAM 2 remaining the default. Phase 1's
findings, especially the reverse-propagation result, shape Phase 2, so it gets its own spec.

## Fairness constraints

- Same chains, same anchor node, same `build_prompts` seed, same `scale`, same `k_max_neg`,
  same multimask and gate settings across all four cells.
- The blow-up guard is a toggle applied identically to both per-slice cells (default off for the
  headline comparison, since it is an add-on, not a model property).
- Scoring uses the identical `merge_metric` and `membrane` code paths for all four, so the numbers
  are comparable to every prior Phase 0, 1, and 2 result.

## Risks and open questions (to resolve during implementation)

1. **Reverse propagation (gating risk).** The pipeline seeds a mid-stack anchor and propagates
   forward and backward. If HF's `propagate_in_video_iterator` cannot run backward from an
   arbitrary frame, SAM 3 propagation is not directly comparable and needs a workaround (dual
   session or frame reindexing). The harness runs the short chain first and fails fast if reverse
   is unavailable, before spending time on the long chain.
2. **Mask-prompt seeding.** The painted-anchor and mid-propagation `add_mask` paths need mask
   prompts. The session state carries `mask_inputs_per_obj`, which suggests support; the adapter
   confirms it. Not needed for the default box-plus-point bake-off, so a gap here does not block
   the headline comparison.
3. **`pred_iou` for SAM 3 propagation.** SAM 3 exposes `iou_scores`, but threading it through
   `PropagationSession`'s SAM 2-specific `_track_step` hook is a Phase 2 interface change. In
   Phase 1 the SAM 3 propagation `pred_iou` may be NaN. This does not affect the merge and membrane
   comparison, which is mask-only.
4. **Memory.** A ~113-frame chain at scale 8 may not fit in 6GB even with offload. The harness
   reports peak VRAM and, on OOM, records the cell as Narval-only rather than failing the whole run.
5. **Coordinate and resolution mapping.** The adapters must return masks in the same pixel space
   and resolution the pipeline functions expect (`_sam`, or `_pcrop` when a crop window is set).
   `post_process_masks` with the frame size as `original_sizes` is the mechanism; this is the most
   error-prone seam and gets a focused unit test.

## Testing

- Pure, torch-free unit tests for the mechanical seams: coordinate passthrough, output-tuple
  shape mapping, threshold and resolution handling, using small synthetic arrays. These run under
  the CPU-only `pytest` suite.
- The adapters themselves need a GPU and the checkpoint, so they are smoke-tested by the harness on
  the short chain, not in CI.

## Success criteria

- The harness produces a four-row table and overlays for at least the short chain, on the local
  6GB card.
- The reverse-propagation question is answered definitively.
- We can state, with merge-metric and membrane numbers, whether SAM 3 beats SAM 2 on propagation,
  on per-slice, or both, which is the input to the Phase 2 decision.
