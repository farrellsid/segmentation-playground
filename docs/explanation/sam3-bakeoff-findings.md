# SAM3 tracker API characterization and bake-off findings

Recorded from `experiments/sam3_probe.py` on 2026-07-21 (RTX 3050 6GB, torch 2.12+cu130,
transformers 5.13.1, checkpoint `F:\sam3\huggingface`). The adapter code in
`sam2_utils/sam3_backend.py` wraps the sequences verified here.

## Load viability (local 6GB card)

Both tracker models load and run on the laptop GPU, with room to spare:

- Image tracker (`Sam3TrackerModel`): peak 2.16 GB.
- Video tracker (`Sam3TrackerVideoModel`, bfloat16): peak 1.37 GB.

So the bake-off runs locally. Nothing here is Narval-only. Loading weights from F: is slow
(config read alone took about 22s), but succeeds. On load, transformers prints a benign notice
that a `sam3_video` checkpoint is instantiating a `sam3_tracker` model; the shared architecture
loads correctly.

## Image tracker (SAM1-style PVS)

Load: `Sam3TrackerModel.from_pretrained(CKPT)` and `Sam3TrackerProcessor.from_pretrained(CKPT)`.

Processor call (pixel coordinates, one image, one object):

```
proc(images=img, input_points=[[[[x, y]]]], input_labels=[[[1]]],
     input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
```

Nesting is verified: `input_points` is `list[list[list[list[float]]]]`, `input_labels` is
`list[list[list[int]]]`, `input_boxes` is `list[list[list[float]]]`. Processor output keys:
`pixel_values, original_sizes, input_points, input_labels`.

Forward: `model(**inputs, multimask_output=True)`. Output keys: `iou_scores, pred_masks,
object_score_logits, image_embeddings`. Shapes: `pred_masks` is `(1, 1, 3, 288, 288)` (batch,
object, num_masks, low-res H, low-res W); `iou_scores` is `(1, 1, 3)`.

Post-process to full resolution:
`proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])` returns a list of
length 1 whose element 0 has shape `(1, 3, 256, 256)` and dtype bool (object, num_masks, H, W),
where 256 was the input size. `post_process_masks(..., binarize=False)` yields logits instead of
bool for the same shapes; `mask_threshold` defaults to 0.0.

Adapter mapping: reshape post-processed masks to `(num_masks, H, W)`, `iou_scores` to
`(num_masks,)`, and use the low-res `pred_masks` (squeezed to `(num_masks, 288, 288)`) as the
`logits` the pipeline's confidence proxy expects.

## Video tracker (SAM2-style PVS)

Load: `Sam3TrackerVideoModel.from_pretrained(CKPT).to(dev, dtype=torch.bfloat16)` and
`Sam3TrackerVideoProcessor.from_pretrained(CKPT)`.

Session init (verified signature, note there is NO `offload_video_to_cpu` kwarg; CPU offload is
via the device kwargs):

```
init_video_session(video, inference_device='cpu', inference_state_device=None,
                   processing_device=None, video_storage_device=None,
                   max_vision_features_cache_size=1, dtype=torch.float32)
```

`video` accepts a list of numpy frames, so the adapter loads the on-disk `{idx:05d}.jpg` frames
into a list and passes them. For a bounded-VRAM run, pass `video_storage_device="cpu"` and
`processing_device="cpu"`.

Add a prompt (mask prompts ARE supported via `input_masks`):

```
add_inputs_to_inference_session(inference_session, frame_idx, obj_ids: list[int] | int,
    input_points=None, input_labels=None, input_boxes=None, input_masks=None,
    original_size=None, clear_old_inputs=True)
```

`clear_old_inputs` has the same polarity as SAM2's `clear_old_points` (True clears), so the
adapter maps them directly. Its default differs (HF True vs SAM2 False), so the adapter passes
the value explicitly.

Propagate (verified signature):

```
propagate_in_video_iterator(inference_session, start_frame_idx=None,
    max_frame_num_to_track=None, reverse=False, show_progress_bar=False)
    -> Iterator[Sam3TrackerVideoSegmentationOutput]
```

## Reverse propagation verdict: SUPPORTED

`reverse=True` is a real kwarg and it worked: propagating with `start_frame_idx=4, reverse=True`
on an 8-frame clip yielded 5 frames (frames 4 down to 0). This clears the plan's gating risk.

One required detail: SAM3 does NOT auto-infer the start frame. Calling
`propagate_in_video_iterator(sess)` with no `start_frame_idx` raises "Cannot determine the
starting frame index; please specify it manually, or run inference on a frame with inputs
first." So the video adapter must remember the annotated (anchor) frame from
`add_inputs_to_inference_session` and pass it as the default `start_frame_idx` for both the
forward and reverse legs. SAM2's `PropagationSession` relies on auto-start, so this defaulting
lives in the adapter.

## Consequences for the adapters

- Image adapter: straightforward; coordinates are pixel-space, matching the pipeline.
- Video adapter: track the anchor frame for `start_frame_idx`; map `clear_old_points ->
  clear_old_inputs`; load frames from the dir into `video=`; yield frame-resolution logits from
  `post_process_masks(..., binarize=False)` so the pipeline's `> 0.0` threshold reproduces the mask.
- `pred_iou` for propagation: the output carries per-object scores, but wiring them through
  `PropagationSession`'s SAM2-specific `_track_step` hook is deferred; propagation `pred_iou` may
  be NaN in this round, which does not affect the mask-only merge and membrane scoring.

## Bake-off results

To be filled in by the harness (Task 6): a four-row table over
{SAM2, SAM3} x {propagation, per-slice} on AIAL chain_05 and chain_00.
