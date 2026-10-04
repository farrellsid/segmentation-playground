# Multi-neuron 2D mask viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user open several of Lucinda's corrected neurons together in napari, each toggleable on/off, composited per z-slice, so they can inspect them as a group instead of one at a time.

**Architecture:** `render_review.py`'s existing `neuron_volume()` already remaps every chain of a neuron onto the shared `_sam`/z grid and crops it to that neuron's own bounding box; it is extended to also return that bounding box's origin (currently computed, then discarded). A new small script, `view_neurons.py`, calls it once per requested neuron, assigns each neuron a distinct small integer ID (multiplied into its volume so napari's own per-label colormap tells neurons apart), converts each neuron's voxel-space origin into real-world nanometers, and adds one napari `Labels` layer per neuron with that `translate`/`scale`, so toggling a neuron is napari's own layer-visibility checkbox.

**Tech Stack:** Python, numpy, napari (already a dependency via `gui.py`), pytest.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (repo-wide rule, `CLAUDE.md`).
- Run the `humanizer` skill on any committed prose (docs, commit messages) before committing it.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. The new test file must not import napari or torch; only a pure, napari-free function is unit tested, matching how `gui.py`'s `_peek_chain` is tested while the real viewer wiring around it is not.
- Lint only the files touched: `ruff check <file>`, never a whole-tree `ruff format`.
- `neuron_volume()`'s new return is `(vol, origin, spacing)`, where `origin` is `(z, y, x)` in voxel units (the same units `vol`'s own axes are in), `(0, 0, 0)` for the empty-volume case. `spacing` and `vol`'s meaning are unchanged from today.
- No merged, monolithic multi-neuron array: each neuron is its own napari `Labels` layer, positioned via `translate`.
- Each neuron's ID-multiplied volume must not silently overflow its dtype for a reasonably large neuron count (use a dtype wider than `uint8` for the multiply).
- `translate` passed to `add_labels` must be in real-world nanometers (`origin` converted via `spacing`), not raw voxel indices, since `scale=spacing` is also passed on the same layer; napari composes `scale` and `translate` in world units.
- No neuron-picker GUI for v1: `view_neurons.py` takes `--tree` and `--neurons` (comma-separated) CLI flags only.
- No VAST-style export or on-disk cache of a merged labelmap, and no custom per-neuron color assignment beyond napari's own default colormap (both explicitly cut in the spec).

Spec: [docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../specs/2026-10-04-multi-neuron-2d-viewer-design.md).

---

### Task 1: `neuron_volume()` returns its origin too

**Files:**
- Modify: `render_review.py:91-148` (`neuron_volume`'s two `return` statements and docstring), `render_review.py:154` (`neuron_mesh`'s call site)
- Modify: `tests/test_render_review_mesh.py` (7 call sites that unpack `neuron_volume()`'s return)

**Interfaces:**
- Consumes: nothing new; `pipeline.chain_masks_in_sam` (existing, unchanged).
- Produces: `neuron_volume(root, neuron, *, max_voxels=400_000_000) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]`, i.e. `(vol, origin, spacing)`. Task 2 depends on this exact 3-tuple shape and on `origin` being `(z, y, x)` voxel offsets.

- [ ] **Step 1: Extend the one test that already describes the origin, to assert it**

In `tests/test_render_review_mesh.py`, replace the `test_mask_lands_at_the_correct_coordinates` test (currently lines 132-146):

Old:
```python
    def test_mask_lands_at_the_correct_coordinates(self, tmp_path, monkeypatch):
        root, placements = _tree_with_known_placement(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, _ = render_review.neuron_volume(root, "AIBL")
        # bbox origin is the min over both blocks: x0=50, y0=280
        # chain 0 (4x10 at x0=50,y0=300) lands at rows 20:24, cols 0:10
        assert vol[0, 20, 0] == 1, "chain 0 top-left corner must be set"
        assert vol[0, 23, 9] == 1, "chain 0 bottom-right corner must be set"
        assert vol[0, 20, 10] == 0, "one column past chain 0's block must be empty"
        assert vol[0, 19, 0] == 0, "one row above chain 0's block must be empty"
        # chain 1 (3x3 at x0=70,y0=280) lands at rows 0:3, cols 20:23
        assert vol[0, 0, 20] == 1, "chain 1 top-left corner must be set"
        assert vol[0, 2, 22] == 1, "chain 1 bottom-right corner must be set"
        assert vol[0, 0, 19] == 0, "one column before chain 1's block must be empty"
        assert vol[0, 3, 20] == 0, "one row below chain 1's block must be empty"
```

New:
```python
    def test_mask_lands_at_the_correct_coordinates(self, tmp_path, monkeypatch):
        root, placements = _tree_with_known_placement(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, origin, _spacing = render_review.neuron_volume(root, "AIBL")
        # bbox origin is the min over both blocks: z=2000, x0=50, y0=280
        assert origin == (2000, 280, 50), f"expected origin (2000, 280, 50), got {origin}"
        # chain 0 (4x10 at x0=50,y0=300) lands at rows 20:24, cols 0:10
        assert vol[0, 20, 0] == 1, "chain 0 top-left corner must be set"
        assert vol[0, 23, 9] == 1, "chain 0 bottom-right corner must be set"
        assert vol[0, 20, 10] == 0, "one column past chain 0's block must be empty"
        assert vol[0, 19, 0] == 0, "one row above chain 0's block must be empty"
        # chain 1 (3x3 at x0=70,y0=280) lands at rows 0:3, cols 20:23
        assert vol[0, 0, 20] == 1, "chain 1 top-left corner must be set"
        assert vol[0, 2, 22] == 1, "chain 1 bottom-right corner must be set"
        assert vol[0, 0, 19] == 0, "one column before chain 1's block must be empty"
        assert vol[0, 3, 20] == 0, "one row below chain 1's block must be empty"
```

(The `(2000, 280, 50)` comes from `_tree_with_known_placement`, already in this file: `z = 2000`, and the two blocks are at `(x0=50, y0=300)` and `(x0=70, y0=280)`, so the min over both is `x0=50, y0=280`.)

- [ ] **Step 2: Run it, confirm it fails for the right reason**

Run: `py -3 -m pytest tests/test_render_review_mesh.py::TestNeuronVolume::test_mask_lands_at_the_correct_coordinates -v`
Expected: FAIL with `ValueError: too many values to unpack (expected 2)` (today's `neuron_volume` still returns a 2-tuple).

- [ ] **Step 3: Extend `neuron_volume()` to return its origin**

In `render_review.py`, replace (lines 91-148):

Old:
```python
def neuron_volume(root, neuron: str, *, max_voxels: int = 400_000_000):
    """Every chain of ``neuron`` composited onto one grid, cropped to its bbox.

    Cropping is what keeps this laptop-sized. The full `_sam` grid across a long
    neuron runs to hundreds of MB, while the neuron's own bounding box is a small
    fraction of that. Returns ``(vol, spacing)`` with ``vol`` as ``(z, y, x)`` uint8.
    """
    root = Path(root)
    blocks = []
    for _ci, cdir in _chain_dirs(root, neuron):
        for z, (mask, x0, y0) in pipeline.chain_masks_in_sam(cdir).items():
            mask = np.asarray(mask, dtype=bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            # Crop each mask to its own CONTENT before anything sizes a bbox from it.
            # chain_masks_in_sam hands back a crop-sized array for a tier-2 `_pcrop`
            # chain but a WHOLE-FRAME array at x0=y0=0 for a legacy `_sam` one, so
            # measuring the array instead of the mask lets one legacy chain size the
            # entire volume. Real case from the AUA bundle: AUAL/chain_00 carries an
            # 84x183 blob on a 1154x1152 array, which pushed the volume to
            # (336, 1152, 1154), 446 million voxels, past the memory budget, so the
            # mesh was never produced at all.
            iy0, iy1 = int(ys.min()), int(ys.max()) + 1
            ix0, ix1 = int(xs.min()), int(xs.max()) + 1
            blocks.append((int(z), mask[iy0:iy1, ix0:ix1],
                           int(x0) + ix0, int(y0) + iy0))
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.uint8), SPACING_SAM8_NM

    # The volume spans every z from the minimum to the maximum present, not just the
    # z values that happen to survive the `mask.any()` filter above. Compacting past
    # a missing z (mask dropout is real here, measured at 5-30% over propagation
    # distance) would pull the surviving planes together and silently fuse a real gap
    # to 50nm, which is exactly the defect this mesh exists to reveal. A missing z
    # instead leaves an empty plane, so a real gap in the source data shows up as a
    # real gap in the mesh.
    z_min = min(b[0] for b in blocks)
    z_max = max(b[0] for b in blocks)
    n_z = z_max - z_min + 1
    x0 = min(b[2] for b in blocks)
    y0 = min(b[3] for b in blocks)
    x1 = max(b[2] + b[1].shape[1] for b in blocks)
    y1 = max(b[3] + b[1].shape[0] for b in blocks)
    shape = (n_z, y1 - y0, x1 - x0)
    n_vox = shape[0] * shape[1] * shape[2]
    if n_vox > max_voxels:
        raise SystemExit(
            f"[render] {neuron} needs a {shape} volume, {n_vox:,} voxels, over the "
            f"{max_voxels:,} budget. Render fewer chains, or raise --max-voxels if "
            f"this machine has the memory.")

    vol = np.zeros(shape, dtype=np.uint8)
    for z, mask, bx, by in blocks:
        h, w = mask.shape
        sy, sx = by - y0, bx - x0
        vol[z - z_min, sy:sy + h, sx:sx + w] |= mask.astype(np.uint8)
    return vol, SPACING_SAM8_NM
```

New:
```python
def neuron_volume(root, neuron: str, *, max_voxels: int = 400_000_000):
    """Every chain of ``neuron`` composited onto one grid, cropped to its bbox.

    Cropping is what keeps this laptop-sized. The full `_sam` grid across a long
    neuron runs to hundreds of MB, while the neuron's own bounding box is a small
    fraction of that. Returns ``(vol, origin, spacing)``: ``vol`` as ``(z, y, x)``
    uint8, ``origin`` as the ``(z, y, x)`` voxel offset of ``vol``'s ``[0, 0, 0]``
    corner in the shared `_sam`/z coordinate space (``(0, 0, 0)`` when ``vol`` is
    empty, since there is nothing to place).
    """
    root = Path(root)
    blocks = []
    for _ci, cdir in _chain_dirs(root, neuron):
        for z, (mask, x0, y0) in pipeline.chain_masks_in_sam(cdir).items():
            mask = np.asarray(mask, dtype=bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            # Crop each mask to its own CONTENT before anything sizes a bbox from it.
            # chain_masks_in_sam hands back a crop-sized array for a tier-2 `_pcrop`
            # chain but a WHOLE-FRAME array at x0=y0=0 for a legacy `_sam` one, so
            # measuring the array instead of the mask lets one legacy chain size the
            # entire volume. Real case from the AUA bundle: AUAL/chain_00 carries an
            # 84x183 blob on a 1154x1152 array, which pushed the volume to
            # (336, 1152, 1154), 446 million voxels, past the memory budget, so the
            # mesh was never produced at all.
            iy0, iy1 = int(ys.min()), int(ys.max()) + 1
            ix0, ix1 = int(xs.min()), int(xs.max()) + 1
            blocks.append((int(z), mask[iy0:iy1, ix0:ix1],
                           int(x0) + ix0, int(y0) + iy0))
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0), SPACING_SAM8_NM

    # The volume spans every z from the minimum to the maximum present, not just the
    # z values that happen to survive the `mask.any()` filter above. Compacting past
    # a missing z (mask dropout is real here, measured at 5-30% over propagation
    # distance) would pull the surviving planes together and silently fuse a real gap
    # to 50nm, which is exactly the defect this mesh exists to reveal. A missing z
    # instead leaves an empty plane, so a real gap in the source data shows up as a
    # real gap in the mesh.
    z_min = min(b[0] for b in blocks)
    z_max = max(b[0] for b in blocks)
    n_z = z_max - z_min + 1
    x0 = min(b[2] for b in blocks)
    y0 = min(b[3] for b in blocks)
    x1 = max(b[2] + b[1].shape[1] for b in blocks)
    y1 = max(b[3] + b[1].shape[0] for b in blocks)
    shape = (n_z, y1 - y0, x1 - x0)
    n_vox = shape[0] * shape[1] * shape[2]
    if n_vox > max_voxels:
        raise SystemExit(
            f"[render] {neuron} needs a {shape} volume, {n_vox:,} voxels, over the "
            f"{max_voxels:,} budget. Render fewer chains, or raise --max-voxels if "
            f"this machine has the memory.")

    vol = np.zeros(shape, dtype=np.uint8)
    for z, mask, bx, by in blocks:
        h, w = mask.shape
        sy, sx = by - y0, bx - x0
        vol[z - z_min, sy:sy + h, sx:sx + w] |= mask.astype(np.uint8)
    return vol, (z_min, y0, x0), SPACING_SAM8_NM
```

- [ ] **Step 4: Fix `neuron_mesh()`'s call site**

In `render_review.py`, replace (line 154):

Old:
```python
    vol, spacing = neuron_volume(root, neuron, max_voxels=max_voxels)
```

New:
```python
    vol, _origin, spacing = neuron_volume(root, neuron, max_voxels=max_voxels)
```

- [ ] **Step 5: Fix the other 6 test call sites**

In `tests/test_render_review_mesh.py`, apply these 6 one-line replacements (each is a tuple-unpack of `neuron_volume()`'s return; each needs one more placeholder for the new middle element):

Line 103, old: `        vol, spacing = render_review.neuron_volume(root, "AIBL")`
new: `        vol, _origin, spacing = render_review.neuron_volume(root, "AIBL")`

Line 113, old: `        vol, _ = render_review.neuron_volume(root, "AIBL")`
new: `        vol, _origin, _ = render_review.neuron_volume(root, "AIBL")`

Line 121, old: `        _vol, spacing = render_review.neuron_volume(root, "AIBL")`
new: `        _vol, _origin, spacing = render_review.neuron_volume(root, "AIBL")`

Line 135 is handled by Step 1 above (that's the test being extended), skip it here.

Line 152, old: `        vol, _ = render_review.neuron_volume(root, "AIBL")`
new: `        vol, _origin, _ = render_review.neuron_volume(root, "AIBL")`

Line 222, old: `        vol, _spacing = render_review.neuron_volume(root, "AUAL")`
new: `        vol, _origin, _spacing = render_review.neuron_volume(root, "AUAL")`

Line 232, old: `        vol, _spacing = render_review.neuron_volume(root, "AUAL")`
new: `        vol, _origin, _spacing = render_review.neuron_volume(root, "AUAL")`

(Line numbers are from before Step 1's edit; re-find each by its old text, not by number, since Step 1 already changed the file's line count.)

- [ ] **Step 6: Run the whole file, confirm everything passes**

Run: `py -3 -m pytest tests/test_render_review_mesh.py -v`
Expected: PASS, all tests in the file (including `test_mask_lands_at_the_correct_coordinates` from Step 1).

- [ ] **Step 7: Run the full suite, to catch any other caller Step 0's grep might have missed**

Run: `py -3 -m pytest -v`
Expected: PASS (same pass count as the repo's current baseline, no new failures).

- [ ] **Step 8: Lint the touched files**

Run: `ruff check render_review.py tests/test_render_review_mesh.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 9: Commit**

```bash
git add render_review.py tests/test_render_review_mesh.py
git commit -m "render: neuron_volume returns its bbox origin, not just the volume

Needed by the upcoming multi-neuron viewer to place each neuron's cropped
volume at its real position in the shared _sam/z grid via napari's translate.
neuron_volume computed this origin internally already and discarded it."
```

---

### Task 2: `view_neurons.py`, the multi-neuron viewer

**Files:**
- Create: `view_neurons.py` (repo root)
- Create: `tests/test_view_neurons.py`
- Modify: `docs/reference/cli.md` (new `## view_neurons.py` section, matching the existing per-script format)

**Interfaces:**
- Consumes: `render_review.neuron_volume(root, neuron) -> (vol, origin, spacing)` from Task 1, exactly that 3-tuple shape.
- Produces: `build_layer_specs(root, neurons: list[str]) -> list[dict]`, each dict `{"name": str, "data": np.ndarray (uint16), "translate": tuple[float, float, float], "scale": tuple[float, float, float]}`. `main()`, the CLI entry point, is not covered by tests (it imports and calls real napari).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_view_neurons.py`:

```python
"""build_layer_specs: per-neuron napari layer specs from a tree or bundle root.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, build_layer_specs here) is tested directly; the real napari
wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import view_neurons


def _fake_neuron_volume(volumes):
    def fake(root, neuron, **kw):
        return volumes[neuron]
    return fake


def test_assigns_distinct_ids_in_selection_order(monkeypatch):
    vol_a = np.ones((2, 3, 3), dtype=np.uint8)
    vol_b = np.ones((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol_a, (100, 0, 0), (50.0, 128.0, 128.0)),
        "AIAR": (vol_b, (100, 0, 0), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["AIAL", "AIAR"])

    assert [s["name"] for s in specs] == ["AIAL", "AIAR"]
    assert set(np.unique(specs[0]["data"]).tolist()) == {0, 1}, (
        "AIAL (selected first) must be label ID 1")
    assert set(np.unique(specs[1]["data"]).tolist()) == {0, 2}, (
        "AIAR (selected second) must be label ID 2, not the same ID as AIAL")


def test_translate_is_origin_converted_to_real_world_nm(monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (10, 20, 30), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["AIAL"])

    assert specs[0]["translate"] == (500.0, 2560.0, 3840.0), (
        "translate must be origin * spacing elementwise (z=10*50, y=20*128, x=30*128), "
        "in real-world nm, not raw voxel indices, since scale=spacing is set too")
    assert specs[0]["scale"] == (50.0, 128.0, 128.0)


def test_a_neuron_with_no_masks_is_skipped_not_an_empty_layer(monkeypatch):
    empty = np.zeros((0, 0, 0), dtype=np.uint8)
    has_masks = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "EMPTYN": (empty, (0, 0, 0), (50.0, 128.0, 128.0)),
        "AIAL": (has_masks, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["EMPTYN", "AIAL"])

    assert [s["name"] for s in specs] == ["AIAL"], (
        "a neuron with no masks must be skipped entirely, not added as an empty layer")


def test_id_volume_does_not_overflow_for_many_neurons(monkeypatch):
    # 300 neurons would overflow uint8 (max 255) if the ID multiply stayed narrow.
    volumes = {f"N{i}": (np.ones((1, 1, 1), dtype=np.uint8), (0, 0, 0),
                         (50.0, 128.0, 128.0)) for i in range(300)}
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume",
                        _fake_neuron_volume(volumes))

    specs = view_neurons.build_layer_specs("fake-root", list(volumes))

    assert specs[-1]["data"].max() == 300, (
        "the 300th neuron's label ID must survive as 300, not wrap around a narrow dtype")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'view_neurons'`.

- [ ] **Step 3: Write `view_neurons.py`**

Create `view_neurons.py`:

```python
"""Open several corrected neurons together in napari, each toggleable on/off.

One napari Labels layer per neuron, each positioned by its own translate/scale
so they land correctly in the shared real-world (nanometre) coordinate space,
reusing render_review.neuron_volume unmodified for the per-neuron remap/crop.
Overlapping neurons are shown blended by napari's own layer compositing; this
tool does not resolve overlaps into one canonical label (see
docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review


def build_layer_specs(root, neurons: list[str]) -> list[dict]:
    """One layer spec per neuron that actually has masks, in selection order.

    A neuron with an empty volume (no masks found under `root`) is skipped
    entirely, not added as an empty layer. Each spec's `data` is `vol` (from
    `render_review.neuron_volume`) multiplied by that neuron's 1-based selection
    index, so napari's own per-label-value colormap gives every neuron a
    distinct, consistent color with no custom palette. The multiply uses uint16,
    wide enough that a neuron count in the hundreds cannot wrap around.

    `translate` is each neuron's voxel-space `origin` converted to real-world
    nanometres via `spacing` (origin * spacing, elementwise), not the raw voxel
    offset, because `scale=spacing` is also set on the same layer and napari
    composes scale and translate in world units.
    """
    specs = []
    for i, neuron in enumerate(neurons, start=1):
        vol, origin, spacing = render_review.neuron_volume(root, neuron)
        if not vol.size or not vol.any():
            print(f"[view] {neuron}: no masks, skipping")
            continue
        id_vol = vol.astype(np.uint16) * np.uint16(i)
        translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))
        specs.append({"name": neuron, "data": id_vol, "translate": translate,
                      "scale": tuple(float(s) for s in spacing)})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D mask viewer (napari)")
    ap.add_argument("--tree", required=True,
                    help="bundle folder or output tree root (same layout either way)")
    ap.add_argument("--neurons", required=True,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL")
    args = ap.parse_args()

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    specs = build_layer_specs(args.tree, neurons)
    if not specs:
        raise SystemExit(
            f"[view] none of {neurons} have any masks under {args.tree}")

    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    print(f"[view] opened {len(specs)} neuron(s): {', '.join(s['name'] for s in specs)}")
    napari.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: PASS, all four tests.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same count as the end of Task 1, plus the 4 new tests).

- [ ] **Step 6: Document the new script**

In `docs/reference/cli.md`, add a new section. Insert it after the `## gui_neuron.py` section (after its flag table, before the next `## eval.score_batch` heading), matching that section's exact format:

```markdown
## view_neurons.py

Opens several corrected neurons together in one napari window, each as its own
toggleable Labels layer, composited per z-slice in the shared real-world coordinate
space. Reuses `render_review.neuron_volume` for the per-neuron remap/crop; does not
resolve overlaps between neurons into one canonical label. See
[../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

| Flag | Effect |
|------|--------|
| `--tree <dir>` | Bundle folder or output tree root to read neurons from (same per-chain layout either way). |
| `--neurons <names>` | Comma-separated neuron names to open together, e.g. `AVAL,AVAR,RIBL`. |
```

Now invoke the `humanizer` skill on this new `cli.md` section and apply any fix it suggests before committing.

- [ ] **Step 7: Lint the touched files**

Run: `ruff check view_neurons.py tests/test_view_neurons.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 8: Manual smoke check against a real bundle (optional but recommended before treating this as done)**

This needs a real bundle and a display, so it is not part of the automated test suite; skip it if neither is available right now, but run it before relying on this in practice. Two real corrected bundles are available for this: `C:\Users\User\Downloads\AIA.zip` and `C:\Users\User\Downloads\AIB.zip`. Extract one (e.g. `AIA.zip` to a scratch folder, giving `<scratch>\AIA\AIAL`, `<scratch>\AIA\AIAR`), then:

```
py -3 view_neurons.py --tree "<scratch>\AIA" --neurons AIAL,AIAR
```

Expected: a napari window opens with two Labels layers, `AIAL` and `AIAR`, each a different color, each independently toggleable in the layers panel, both showing real mask content when the z-slider is moved into their range.

- [ ] **Step 9: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py docs/reference/cli.md
git commit -m "view: multi-neuron 2D mask viewer in napari

Opens several neurons together as separate, toggleable Labels layers,
positioned in shared real-world nm coordinates via render_review.neuron_volume's
new origin return (previous commit). Reuses that function unmodified; this
commit only adds the per-neuron ID assignment and the napari wiring around it.

3D mesh view, contact mapping, and a neuron-picker GUI are deferred, not built
here: docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md."
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** "Reusing neuron_volume() unmodified" in the spec turned out to be inaccurate once the origin requirement was discovered; the user chose extending the return tuple over duplicating the remap logic (see the spec's note on this and Task 1 here). Every other spec section maps directly to Task 2: multi-neuron compositing via translated layers, toggle as native layer visibility, overlaps shown not resolved (the module docstring says so explicitly, no arbitration code is written), the `view_neurons.py` entry point and its exact CLI shape, and the pure-function test strategy.
- **Placeholder scan:** no TBD/TODO/"handle edge cases" left in either task; both test files have concrete, complete code.
- **Type consistency:** `neuron_volume()`'s new 3-tuple `(vol, origin, spacing)` from Task 1 is consumed identically in Task 2's `build_layer_specs` (`vol, origin, spacing = render_review.neuron_volume(root, neuron)`), and the test fakes in `tests/test_view_neurons.py` return exactly that 3-tuple shape to match.
