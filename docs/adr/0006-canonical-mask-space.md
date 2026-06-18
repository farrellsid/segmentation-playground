# 0006. Canonical mask space and encoding

Status: Accepted

## Context

Masks can be stored at different resolutions and in different encodings. An early version mixed these
and produced masks that looked empty (a uint16 instance label of value 1 is invisible and is
destroyed by a 16-bit to 8-bit conversion) and a 2x skeleton-containment offset from resampling
between scales. QC needs one fixed answer to "what space and format is a mask on disk."

## Decision

Store masks at the SAM2 grid (`_sam`), with `save_downscale == scale == 8`, so there is no resample.
Write them as 0/255 uint8 single-channel PNGs named `mask_<catmaid_z:04d>.png`. `run_qc` hard-guards
`scale == save_downscale`, skipped only in crop mode, where the window remaps nodes instead.

Keep two mask encodings distinct. `pipeline.save_masks` writes 0/255 uint8 for the single-object
pipeline. `qc.save_masks` writes uint16 instance labels (pixel value equals object id) for the future
multi-object aggregation. They are not interchangeable.

## Consequences

- A saved mask is directly viewable and pixel-comparable to the reference notebook output.
- No resample means no scale-induced offset between a mask and the skeleton nodes used to score it.
- Tier-2 chains are the documented exception: they store masks in `_pcrop` and persist the crop
  window to `state.json` so QC, the viewer, and the GUI can rebuild the space. See
  [0009](0009-tier2-crop-fallback.md) and [coordinate-spaces.md](../reference/coordinate-spaces.md).
- Instance-label encoding waits for multi-object aggregation. Using it in the single-object path was
  the original "empty masks" bug.
