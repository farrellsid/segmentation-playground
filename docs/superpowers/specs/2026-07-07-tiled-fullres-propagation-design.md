# Tiled full-res propagation: the real resolution lever

Status: design, not yet implemented. This is the "with stitch" idea from the resolution
experiments (2026-07-06), scoped as a buildable v1, and the chosen alternative to raising
`image_size` (the bigimg variant, which crashed inside SAM2 and is off-distribution anyway).

## Why

SAM2 resizes every input to a fixed 1024 square, so effective resolution is capped by that
1024, not by GPU memory. The tier-2 crop already exploits this: cropping a small window means
1024 covers a small physical area, so a thin neurite is seen at high effective resolution. The
whole-image variants confirmed the ceiling from the other side (a 9.7k-px frame downsampled to
1024 loses detail; scale 1 vs scale 4 should look near-identical).

The gap: when a chain's crop is large, the current tier-2 code bumps `crop_scale` coarser so the
input still fits under `chain_crop_max_px` (1536), trading resolution away exactly where the
chain is big. Raising `image_size` to feed more pixels crashes on the pretrained weights (the
bigimg `.view()` error) and is off-distribution regardless. Tiling keeps full resolution over a
large region while staying at the native 1024 input per tile, in-distribution.

## Goal

For a chain whose full-resolution crop exceeds one SAM2 input, split the crop into overlapping
tiles no larger than one input, propagate each tile independently, and union the per-tile masks
back into the chain's crop space. Save and treat the result as a high-resolution tier-2 chain,
so aggregation and the GUI need no changes.

Single object per chain is what makes the stitch simple: there is nothing to disambiguate across
tile borders, overlapping predictions are just OR'd.

## Design

Config knobs on `PipelineConfig` (all off by default, so existing runs are untouched):

- `tile_enable: bool = False`
- `tile_input_px: int = 1024` (target tile input edge; a tile covers `tile_input_px * crop_scale`
  tif px, so at `crop_scale=1` a tile is ~1024 tif px)
- `tile_overlap_frac: float = 0.2` (neighbor overlap, so a neurite crossing a border is seen
  whole in at least one tile)

Flow, per chain, when `tile_enable` and the chain's full-res crop (at `crop_scale=1`) has a
longest edge greater than `tile_input_px`:

1. Build the chain crop window at `crop_scale=1` (full res), the same `chain_crop_window` as
   tier-2 but without the coarsening bump.
2. Tile it: a grid of overlapping sub-windows over the crop, each `<= tile_input_px` on a side,
   stepping by `tile_input_px * (1 - tile_overlap_frac)`. New helper `tile_crop_window` in
   `crop.py` returns the list of sub-`CropWindow`s.
3. Per tile, seed and propagate independently in that tile's crop space: gather the chain's
   skeleton nodes whose `(x, y)` fall inside the tile; pick the tile's anchor as the in-tile
   node nearest the chain's global anchor z; run the normal image-mode prompt (positive node +
   nearest-node negatives) then box, then video-propagate over the chain's z-range. A tile with
   no chain node is skipped (the chain is not there).
4. Stitch: remap each tile's per-frame mask into the chain crop space with the existing
   `remap_mask_to_window` and OR them (single object). Save the unioned masks in the chain crop
   space.
5. Persist the chain crop window (not the tiles) to `state.json`, so the tiling is internal and
   downstream sees an ordinary high-res tier-2 chain.

## Components and isolation

- `crop.py`: `tile_crop_window(cw, *, tile_input_px, overlap_frac) -> list[CropWindow]`, pure
  geometry, unit-testable (coverage of the parent window, overlap, count).
- Orchestrator: a tiled branch that loops tiles through the existing per-tile seed and propagate
  primitives and unions the results. Reuses `remap_mask_to_window` for the stitch.
- No change to aggregation, QC, or the GUI: a tiled chain is saved exactly like a tier-2 chain.

## Testing

- `tile_crop_window`: synthetic windows assert the tiles cover the parent, respect the overlap
  and max size, and reduce to a single tile when the crop already fits. Torch-free.
- Stitch: a pure test that unioning known sub-masks reproduces a known full mask. Torch-free.
- The seed/propagate loop itself needs a GPU, so it is exercised by a smoke run on one large
  chain, then scored against GT, not by a unit test.

## Risks and open questions

- **Per-tile seeding is the hard part.** A tile that the neurite passes through but that has no
  skeleton node on the anchor frame is skipped in v1; the overlap from adjacent node-bearing
  tiles should cover most of the neurite, but a long node-sparse stretch could be dropped. If
  that shows up, v2 seeds a node-less tile from the propagated mask handed off at the shared
  border of an already-run neighbor.
- **Seam over-fill.** Union takes the more inclusive prediction where tiles disagree, so seams
  can over-fill slightly. Acceptable for v1; measured against GT.
- **Cost.** A large chain becomes N propagations. Bounded because tiling triggers only when the
  crop exceeds one input, and only for the chains that need it.
- Connectivity and thin-structure losses (clDice and friends) are out of scope here; this spec is
  only about not throwing resolution away on large chains.

## v1 scope

Tile the tier-2 crop, seed each tile independently from its in-tile nodes, propagate per tile,
union into the chain crop space, save as a high-res tier-2 chain. Gate expansion (border handoff
seeding, seam arbitration) on GT numbers from the v1 smoke.
