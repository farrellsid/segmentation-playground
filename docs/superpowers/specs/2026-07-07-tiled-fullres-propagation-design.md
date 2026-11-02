# Tiled full-res propagation: coarse-to-fine, mask-seeded

Status: design, not yet implemented. Revised after review; supersedes the skeleton-grid
seeding of the first draft. Builds on the mask-seeded-propagation spec (same date) and is the
chosen alternative to raising `image_size` (the bigimg variant, which crashed inside SAM2 and
is off-distribution anyway).

## Why, and where the resolution actually comes from

SAM2 resizes every crop to a fixed 1024 input, so the effective resolution of a chain's
segmentation is `min(crop_tif / crop_scale, 1024) / crop_tif` samples per tif pixel. Two
regimes: read the crop fine enough that its input reaches 1024 and effective resolution is
`1024 / crop_tif`, at which point the read scale stops mattering; read it coarser than 1024 and
SAM2 upsamples, so the input's sample count is the ceiling and reading finer helps.

Measured on the tier2forced run (621 chains): the longest crop edge is a median of 1560 tif px,
and only 14% exceed 2048. Two consequences:

1. The current tier-2 reads at `crop_scale=2`, so the median crop is fed at ~780 input px,
   below 1024. SAM2 upsamples it, so 86% of chains (crop under 2048 tif) under-fill the 1024
   input and leave resolution on the table (about 0.50 samples/tif at the median). Reading the
   same crops at `crop_scale=1` fills the input (1560 downsampled to 1024) and lifts effective
   resolution to about 0.66/tif, roughly a third sharper, for free on the cluster. This is the
   cheap win and is worth shipping on its own.
2. To go past `1024 / crop_tif` you must shrink the physical extent per input, which means
   tiling. A 1560 tif crop split into ~800 tif tiles roughly doubles effective resolution
   (~1.28/tif). Because the median crop already exceeds 1024 tif, tiling helps essentially every
   chain, not just large ones. The open question is whether ~0.66/tif after the scale-1 fix is
   already sharp enough for the neurites, an eyeball-and-GT question, not an assumption.

The first draft also had a real seeding hole. A chain's skeleton is a 1D centerline, so on the
anchor slice it sits at a single point; a grid of tiles would leave almost every tile with no
prompt. The coarse-to-fine design below removes that hole by seeding tiles from a mask, not a
point.

## The building block: scale-1 tier-2 (the cheap win, ship first)

A tile is just a tier-2 crop read fine enough to fill one input. The cheap step, worth shipping
before any tiling, is a scale-1 tier-2 mode: `chain_crop_scale=1` with `chain_crop_max_px`
raised so crops are not coarsened straight back (on the cluster VRAM is not binding). Per the
measurement above this fills SAM2's 1024 input for the 86% of chains whose crop is under 2048
tif, where scale 2 currently under-fills it, so it lifts effective resolution by roughly a
third at essentially no cost. A new preset (`original_tier2_s1`) makes it a one-line cluster
variant alongside the others.

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

The measurement is done (median crop 1560 tif, 14% over 2048), so the order is set:

1. Ship scale-1 tier-2 and mask-seeding first. Both are cheap, and scale-1 alone recovers about
   a third of the effective resolution the current scale-2 crops throw away by under-filling the
   1024 input.
2. Build coarse-to-fine tiling next. The median crop already exceeds 1024 tif, so tiling helps
   essentially every chain; the question is diminishing returns past scale-1, not whether it
   triggers. Judge it against scale-1 tier-2 and the coarse pass on GT.
3. Mask-seeding is orthogonal (it improves the seed, not the resolution) and lands with either.
