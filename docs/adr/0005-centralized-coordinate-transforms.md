# 0005. Centralized coordinate transforms

Status: Accepted

## Context

The pipeline moves between several coordinate systems: CATMAID stack pixels, full-resolution tif
pixels, the downscaled SAM2 grid, two kinds of crop window, CATMAID nanometers, and two z-axis
conventions that differ by an offset. When these conversions are scattered as inline `/ scale` and
`+/- offset` arithmetic, axis swaps and off-by-scale bugs creep in. The prior-art liver pipeline had
a known x/y swap trap of exactly this kind.

## Decision

Put every coordinate transform in `sam2_utils/alignment.py`. Tag every variable with its space
suffix: `_cm`, `_tif`, `_sam`, `_crop`, `_pcrop`. Route all conversions through the module: the
CATMAID-to-tif affine, the resolution maps, the z maps, the nanometer divide, and the `CropWindow`
class that owns the crop math. Isolate the one row/column versus (x, y) swap to a single method,
`CropWindow.slice_tif`.

## Consequences

- "Where is this transform" has one answer. A reader learns the suffixes once and reads geometry
  anywhere in the code.
- The swap lives in one method, guarded by a round-trip test, instead of being re-derived at each
  call site.
- The transforms are unit-tested without a GPU (`tests/test_alignment.py`), so they are safe to
  refactor.
- See [coordinate-spaces.md](../reference/coordinate-spaces.md) for the spaces and the invariants
  that depend on them.
