# Coordinate spaces and invariants

Every coordinate in the pipeline lives in a named space. The conversions between spaces all live in
`sam2_utils/alignment.py`. Tag variables with their space suffix and route conversions through that
module. Do not write `/ scale` or `+/- FILE_Z_OFFSET` inline anywhere else.

## The spaces

| Suffix | Space | Definition |
|--------|-------|------------|
| `_cm` | CATMAID stack pixels | The coordinate system of the CATMAID skeleton nodes. |
| `_tif` | Full-resolution stack pixels | The raw `.tif` image grid. |
| `_sam` | SAM2 input and on-disk mask grid | `_tif / scale` (scale is 8). The space SAM2 propagates in and the canonical space masks are stored in. |
| `_crop` | High-res anchor crop | A window around the anchor node, run at higher resolution for the one-frame seed. |
| `_pcrop` | Per-chain tier-2 crop | A window sized to a whole chain's extent, propagated at higher resolution. |

There is also CATMAID nanometers (the raw node units before the divide to `_cm`), and the z-axis
maps between file-z and CATMAID-z, which differ by `FILE_Z_OFFSET`.

## What lives in alignment.py

- The CATMAID-to-tif affine: `catmaid_to_tif` applies the stored transform; `fit_affine` refits it
  from a landmark set.
- The resolution maps: `tif_to_sam` and `sam_to_tif`.
- The z maps: `catmaid_z_to_file_z` and `file_z_to_catmaid_z`.
- The nanometer divide: `nm_to_stack_px`.
- `CropWindow`, which holds the `_tif` to `_crop` to `_sam` (and `_pcrop`) window math.

## Invariants to respect

**Mask space and filenames.** Masks are 0/255 uint8 single-channel PNGs named
`mask_<catmaid_z:04d>.png`, stored at `_sam` with `save_downscale == scale == 8`. There is no
resample. `run_qc` hard-guards `scale == save_downscale`, skipped only in crop mode, where the window
remaps nodes instead.

**Two mask encodings, do not confuse them.** `pipeline.save_masks` writes 0/255 uint8. `qc.save_masks`
writes uint16 instance labels, where the foreground pixel value equals the object id. The
single-object pipeline uses the former. Instance-label encoding is a multi-object concern.

**The row/column swap is isolated.** The only place that swaps `[y, x]` against `(x, y)` is
`CropWindow.slice_tif`. Keeping it in one method avoids the off-by-axis bugs that come from scattering
the swap through the code.

**Tier-2 chains store masks in `_pcrop`.** They persist their `CropWindow` to `state.json`
(`ChainState.crop_window`), so QC, the read-only viewer, and the GUI all rebuild the crop space.
The containment radius is rescaled by `scale / crop_scale`.

**Skeleton containment uses this chain's nodes.** The `skeleton_contained` QC probe must use the
nodes of the chain being scored, not the whole neuron's. A multi-neuron skeleton's centroid sits off
any single process and would false-flag every frame. The probe is tri-state: True, False, or NaN when
there is no node at that z. Only an explicit False flags.

## If you refit the affine

The affine constants in `sam2_utils/config.py` were fit at one CATMAID section from a set of
landmarks. If you refit on a different section, update `M_AFFINE` and `T_AFFINE` there so every caller
picks up the new values.
