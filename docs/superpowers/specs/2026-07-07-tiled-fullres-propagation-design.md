# Tiled full-res propagation: coarse-to-fine, mask-seeded

Status: design, not yet implemented. Revised after review; supersedes the skeleton-grid
seeding of the first draft. Builds on the mask-seeded-propagation spec (same date) and is the
chosen alternative to raising `image_size` (the bigimg variant, which crashed inside SAM2 and
is off-distribution anyway).

## Why, and the honest limit

SAM2 resizes every input to a fixed 1024 square, so effective resolution is capped by that
1024, not by GPU memory. Cropping beats the cap because a crop makes 1024 cover a small
physical area. Tier-2 already crops to the chain; it only loses resolution when a chain is so
large that even its crop will not fit one input, at which point it reads coarser.

So be clear about the scope: tiling is **not** broadly better than tier-2. For a compact chain
whose crop already fits one input, a tile grid is one tile, identical to tier-2. Tiling earns
its keep only on the minority of large or sprawling chains (AVAL-like) where tier-2 currently
coarsens. Whether that minority is worth the build is a measurement, not an assumption: count
how many chains' full-resolution crop exceeds one input before committing.

The first draft also had a real seeding hole. A chain's skeleton is a 1D centerline, so on the
anchor slice it sits at a single point; a grid of tiles would leave almost every tile with no
prompt. The coarse-to-fine design below removes that hole by seeding tiles from a mask, not a
point.

## The building block: scale-1 tier-2

A tile is just a tier-2 crop read at full resolution that fits one input. So the first, cheap
step is a scale-1 tier-2 mode: `chain_crop_scale=1` with `chain_crop_max_px` raised (on the
cluster VRAM is not binding). That alone is the "original-resolution crop" test, and it tells
us how often a chain exceeds one input (the chains that then coarsen are exactly the tiling
candidates).

## Design: coarse-to-fine

1. **Coarse pass (reused, not repeated).** Run the existing tier-2 (or scale-1 tier-2) pass
   once. Its saved masks are the approximation. Per the mask-seeded spec, a later pass reads
   them via `seed_from="saved_masks"`, so the coarse pass is never re-run unless tier-1 itself
   changes.
2. **Place tiles from the prediction, not the skeleton.** Lay full-resolution tiles over where
   the coarse mask actually is (its per-frame foreground extent plus a margin), only where
   there is mask. No empty tiles by construction, and coverage follows the real cell, not the
   skeleton bbox.
3. **Fine pass, mask-seeded per tile.** Seed each tile with the coarse mask cropped to that
   tile (a mask prompt via `add_new_mask`), then propagate at full resolution in the tile's
   crop space. Because every tile is seeded by the coarse mask it overlaps, there is no
   point-prompt hole.
4. **Stitch.** Remap each tile's per-frame mask into the chain crop space with the existing
   `remap_mask_to_window` and OR them; single object, so overlaps just union. Save in the chain
   crop space and persist the chain crop window (not the tiles) to `state.json`, so downstream
   sees an ordinary high-resolution tier-2 chain.

Tile knobs on `PipelineConfig` (off by default): `tile_enable`, `tile_input_px` (default 1024),
`tile_overlap_frac` (default 0.2).

## Components and isolation

- `crop.py`: `tile_windows_from_mask(coarse_masks, chain_cw, *, tile_input_px, overlap_frac)`
  returns the fine sub-`CropWindow`s covering the coarse foreground. Pure geometry, testable.
- Orchestrator: a tiled branch that, per tile, seeds from the coarse mask (mask-seeded spec) and
  propagates, then unions via `remap_mask_to_window`.
- No change to QC, aggregation, or the GUI; a tiled chain is saved like a tier-2 chain.

## Testing

- `tile_windows_from_mask`: synthetic coarse masks assert tiles cover the foreground, respect
  overlap and max size, and reduce to one tile when the foreground fits one input. Torch-free.
- Stitch: unioning known sub-masks reproduces a known full mask. Torch-free.
- The seed-and-propagate loop needs a GPU, so it is a smoke on one large chain scored against
  GT, not a unit test.

## Risks and open questions

- **Coverage vs the coarse mask's errors.** Tiles follow the coarse mask, so a region the
  coarse pass missed entirely gets no tile. This refines what the coarse pass found; it does not
  discover new structure. Acceptable, since the goal is sharper boundaries on the traced cell.
- **Seam over-fill.** Union takes the more inclusive prediction where tiles overlap, so seams
  can over-fill slightly. Measured against GT.
- **Cost.** A large chain becomes N propagations, bounded because tiling triggers only when the
  crop exceeds one input.

## Sequence

1. Measure the large-chain fraction from existing runs (decides whether tiling is worth it).
2. Ship scale-1 tier-2 (cheap) and mask-seeding (broad) first.
3. Build coarse-to-fine tiling only if the measurement says enough chains coarsen, and judge it
   against scale-1 tier-2 and the coarse pass on GT.
