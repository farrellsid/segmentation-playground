# Lasso-add mask tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a freehand "lasso-add" tool to `gui.py`'s mask correction workflow: draw a loop over the
area to extend, release, and it unions into the current frame's mask, undo-compatible with the
existing paint brush's `Ctrl+Z`.

**Architecture:** A new napari Shapes layer (`self._lasso`) in `add_polygon_lasso` mode, armed by a
new `l` key. A pure, torch/napari-free helper (`gui._rasterize_lasso_fill`) rasterizes a finished
loop's vertices and unions them into a boolean mask; a Shapes `events.data` callback
(`_on_lasso_drawn`) applies that union to the live `Labels` layer via its own `data_setitem` (which
is undo-tracked) and clears the stroke from the lasso layer.

**Tech Stack:** napari 0.7.0 (already installed; `add_polygon_lasso` mode and `Labels.data_setitem`
both confirmed present by reading the installed source), `cv2` (already a project dependency, used
in `pipeline/crop.py`), `numpy`.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (project-wide rule, `CLAUDE.md`).
- `py -3 -m pytest -q` must pass after every task (335 passed, 1 skipped at the start of this plan).
- `ruff check .` must be clean on every file touched.
- Only addition is in scope. No cut/subtract mode, no boundary-arc snapping, no modifier key: the
  design doc (`docs/superpowers/specs/2026-08-12-lasso-add-mask-tool-design.md`) scoped these out
  explicitly, subtraction stays on the existing eraser brush.
- `gui.py` imports `napari`, `torch` (via `pipeline`), and `cv2` lazily (inside functions/methods,
  never at module level), so `import gui` alone stays fast and GPU/display-free for pure-function
  tests. Match this pattern for every new import this plan adds.
- Commit incrementally, one concern per commit (project-wide rule, `CLAUDE.md`).

---

### Task 1: Pure rasterize/union helper

**Files:**
- Modify: `gui.py`
  - Add `_LASSO_EDGE_COLOR` constant, immediately after `_BOX_EDGE_COLOR` (currently `gui.py:108`).
  - Add `_rasterize_lasso_fill`, immediately after `_box_on_frame` (currently ends `gui.py:137`).
- Test: `tests/test_gui_lasso.py` (new)

**Interfaces:**
- Produces: `gui._rasterize_lasso_fill(mask_hw: np.ndarray, polygon_yx) -> np.ndarray`. `mask_hw` is
  any array castable to boolean, shape `(H, W)`. `polygon_yx` is an `(N, 2)` array-like of `(y, x)`
  vertices in the same pixel space as `mask_hw`. Returns a NEW boolean `(H, W)` array (`mask_hw` is
  never mutated): the union of `mask_hw` with the polygon's filled interior, or an unchanged copy of
  `mask_hw` if `polygon_yx` has fewer than 3 vertices. Consumed by Task 2's `_on_lasso_drawn`.
- Produces: `gui._LASSO_EDGE_COLOR` (`str`, a hex color for the new Shapes layer). Consumed by
  Task 2's `_new_lasso_layer`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_lasso.py`:

```python
"""Unit tests for gui._rasterize_lasso_fill, the pure rasterize-and-union step behind
the lasso-add mask tool: draw a freehand loop over the area to extend, release, and
its interior unions into the current frame's mask (see
docs/superpowers/specs/2026-08-12-lasso-add-mask-tool-design.md).

Torch-free / napari-free: gui.py imports torch (via pipeline), napari, and cv2 all
lazily, so importing the module and exercising this module-level helper needs no
GPU, no display, and no viewer (same tactic as test_gui_box.py).

Run either way:
    py -3 -m pytest tests/test_gui_lasso.py
    py -3 tests/test_gui_lasso.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import gui


# ---------------------------------------------------------------------------
# _rasterize_lasso_fill: union correctness, degenerate input, no mutation
# ---------------------------------------------------------------------------

def test_rasterize_lasso_fill_unions_new_area():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True                              # a 3x3 existing mask, area 9
    polygon_yx = np.array([[3, 3], [3, 8], [8, 8], [8, 3]], dtype=float)  # overlaps + extends

    out = gui._rasterize_lasso_fill(mask, polygon_yx)

    assert out.dtype == bool
    assert out.sum() == 41                              # 9 original + 32 newly added
    assert np.all(out[2:5, 2:5])                        # nothing already-mask was removed


def test_rasterize_lasso_fill_degenerate_polygon_is_a_noop():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    polygon_yx = np.array([[3, 3], [3, 4]], dtype=float)  # 2 vertices, not a real loop

    out = gui._rasterize_lasso_fill(mask, polygon_yx)

    assert np.array_equal(out, mask)


def test_rasterize_lasso_fill_does_not_mutate_input():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    before = mask.copy()
    polygon_yx = np.array([[3, 3], [3, 8], [8, 8], [8, 3]], dtype=float)

    gui._rasterize_lasso_fill(mask, polygon_yx)

    assert np.array_equal(mask, before)


# ---------------------------------------------------------------------------
# plain runner
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_gui_lasso.py -v`
Expected: FAIL, `AttributeError: module 'gui' has no attribute '_rasterize_lasso_fill'`

- [ ] **Step 3: Add the color constant and the pure helper**

In `gui.py`, immediately after the `_BOX_EDGE_COLOR` line (`gui.py:108`):

```python
_BOX_EDGE_COLOR = "#1f77b4"   # blue, distinct from the green/red prompt points
_LASSO_EDGE_COLOR = "#ff7f0e"   # orange, distinct from box (blue) and prompts (green/red)
```

Immediately after `_box_on_frame` (`gui.py:128-137`), add:

```python
def _rasterize_lasso_fill(mask_hw: np.ndarray, polygon_yx) -> np.ndarray:
    """Union a freehand lasso polygon's interior into a single-frame (H, W) mask.

    ``polygon_yx`` is an (N, 2) array of (y, x) vertices in the same pixel space as
    ``mask_hw`` (matching how prompts/box vertices already round-trip in this file).
    Returns a NEW boolean array; ``mask_hw`` is never mutated. A degenerate polygon
    (fewer than 3 vertices) returns a copy of ``mask_hw`` unchanged: napari's own
    add_polygon_lasso mode already drops anything that small before this is ever
    called from _on_lasso_drawn (confirmed against the installed napari 0.7.0
    source, Shapes._finish_drawing), so this guard only matters for a direct call,
    e.g. from a test."""
    import cv2
    mask_hw = np.asarray(mask_hw, dtype=bool)
    polygon_yx = np.asarray(polygon_yx, dtype=float)
    if len(polygon_yx) < 3:
        return mask_hw.copy()
    poly_xy = np.round(polygon_yx[:, ::-1]).astype(np.int32)   # cv2 wants (x, y)
    filled = np.zeros(mask_hw.shape, dtype=np.uint8)
    cv2.fillPoly(filled, [poly_xy], 1)
    return mask_hw | filled.astype(bool)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3 -m pytest tests/test_gui_lasso.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite and lint**

Run: `py -3 -m pytest -q`
Expected: `338 passed, 1 skipped` (335 passed, 1 skipped before this task, plus this task's 3 new tests)

Run: `ruff check gui.py tests/test_gui_lasso.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add gui.py tests/test_gui_lasso.py
git commit -m "Add pure rasterize/union helper for the lasso-add mask tool"
```

---

### Task 2: Wire the lasso tool into the live GUI

**Files:**
- Modify: `gui.py`
  - `ReviewGUI.__init__`, the layer-attribute None-init line (currently `gui.py:381`).
  - `ReviewGUI.open_chain`, where `self._box` is built (currently `gui.py:480-483`).
  - New method `_new_lasso_layer`, immediately after `_new_box_layer` (currently ends `gui.py:524`).
  - New method `_on_lasso_drawn`, immediately after `_new_lasso_layer`.
  - New method `activate_lasso_draw`, immediately after `activate_box_draw` (currently `gui.py:619-627`).
  - `ReviewGUI._bind_keys`, immediately after the `b` binding (currently `gui.py:1298-1299`).
  - Module docstring, line 7 (mentions "paint an anchor mask"; add the lasso tool alongside it).
- Modify: `docs/CHANGELOG.md` (new dated entry).

**Interfaces:**
- Consumes: `gui._rasterize_lasso_fill`, `gui._LASSO_EDGE_COLOR` (Task 1); `self._mask.data_setitem`
  (napari `Labels` API, confirmed present on the installed version, writes through undo history);
  `self.data.obj_id` (existing `ReviewData` field); `self._current_frame()` (existing method,
  `gui.py:616-617`).
- Produces: `ReviewGUI._lasso` (a napari `Shapes` layer, `None` until a chain is open),
  `ReviewGUI._new_lasso_layer(self, scale=(1.0, 1.0, 1.0))`, `ReviewGUI._on_lasso_drawn(self,
  event=None) -> None`, `ReviewGUI.activate_lasso_draw(self, *_) -> None`, the `l` keybind.

- [ ] **Step 1: Add `self._lasso` to the layer-attribute init**

In `gui.py`, `ReviewGUI.__init__` (`gui.py:381`), change:

```python
        self._img = self._mask = self._skel = self._prompts = self._box = None
```

to:

```python
        self._img = self._mask = self._skel = self._prompts = self._box = self._lasso = None
```

- [ ] **Step 2: Add `_new_lasso_layer`**

In `gui.py`, immediately after `_new_box_layer` (which currently ends at `gui.py:524` with `return
layer`, right before `def _seed_prompts_from_state`), add:

```python
    def _new_lasso_layer(self, scale=(1.0, 1.0, 1.0)):
        """An empty Shapes layer for freehand loop-and-union mask additions (the 'L'
        key arms add_polygon_lasso mode for the next stroke). ndim=3 so a loop binds to
        a specific frame (t, y, x); ``scale`` matches the other layers so a drawn loop's
        data coords stay _sam (or _pcrop), the same convention _box/_prompts already
        use. Each finished stroke is consumed immediately by _on_lasso_drawn (unioned
        into the mask, then removed from this layer), so it never accumulates drawn
        loops on screen."""
        layer = self.viewer.add_shapes(
            name="lasso", ndim=3, scale=scale, edge_color=_LASSO_EDGE_COLOR,
            face_color="transparent", edge_width=1.0, opacity=0.8)
        layer.events.data.connect(self._on_lasso_drawn)
        return layer
```

- [ ] **Step 3: Add `_on_lasso_drawn`**

Immediately after `_new_lasso_layer`, add:

```python
    def _on_lasso_drawn(self, event=None) -> None:
        """napari Shapes 'data' event handler for self._lasso: on a freshly finished
        freehand loop (event.action == ActionType.ADDED, napari's own signal that a
        shape was just completed, not edited or removed), rasterize it and union it
        into the current frame's mask via Labels.data_setitem (undo-tracked, so
        Ctrl+Z reverts one lasso stroke exactly like it already reverts one paint
        stroke), then remove the consumed shape from self._lasso so it stays empty
        between strokes.

        Filtering on ActionType.ADDED (rather than a re-entrancy flag) is what keeps
        this callback from re-triggering on its own trim below: removing the last
        shape fires REMOVING then REMOVED, never ADDED (confirmed against the
        installed napari 0.7.0's Shapes.data setter directly), so that self-write is
        naturally ignored rather than needing a guard flag."""
        from napari.layers.base import ActionType
        if event is not None and getattr(event, "action", None) != ActionType.ADDED:
            return
        if self._lasso is None or self._mask is None or self.data is None:
            return
        shapes_data = self._lasso.data
        if not len(shapes_data):
            return
        verts = np.asarray(shapes_data[-1], dtype=float)
        frame_idx = self._current_frame()
        on_frame = np.round(verts[:, 0]).astype(int) == frame_idx
        polygon_yx = verts[on_frame][:, 1:]                    # drop t -> (y, x)
        if len(polygon_yx) >= 3:
            mask_hw = np.asarray(self._mask.data[frame_idx] == self.data.obj_id)
            new_mask = _rasterize_lasso_fill(mask_hw, polygon_yx)
            added = new_mask & ~mask_hw
            ys, xs = np.nonzero(added)
            if len(ys):
                ts = np.full(len(ys), frame_idx, dtype=int)
                self._mask.data_setitem((ts, ys, xs), self.data.obj_id)
        self._lasso.data = shapes_data[:-1]
```

- [ ] **Step 4: Add `activate_lasso_draw`**

In `gui.py`, immediately after `activate_box_draw` (currently `gui.py:619-627`), add:

```python
    def activate_lasso_draw(self, *_) -> None:
        """Make the lasso layer active in add_polygon_lasso mode, so the next
        freehand drag draws a loop that unions into the mask on release (the 'L'
        key)."""
        if self._lasso is None:
            print("[gui] no chain open; nothing to draw a lasso on")
            return
        self.viewer.layers.selection.active = self._lasso
        self._lasso.mode = "add_polygon_lasso"
        print("[gui] lasso mode: drag a loop over the area to add, release to commit")
```

- [ ] **Step 5: Build `self._lasso` in `open_chain`**

In `gui.py`, `ReviewGUI.open_chain` (`gui.py:480-483`), change:

```python
        self._prompts = self._new_prompts_layer(scale=lscale)
        self._box = self._new_box_layer(scale=lscale)
        # pre-load the chain's original seed (points + box) at the anchor frame
        self._seed_prompts_from_state()
```

to:

```python
        self._prompts = self._new_prompts_layer(scale=lscale)
        self._box = self._new_box_layer(scale=lscale)
        self._lasso = self._new_lasso_layer(scale=lscale)
        # pre-load the chain's original seed (points + box) at the anchor frame
        self._seed_prompts_from_state()
```

- [ ] **Step 6: Bind the 'l' key**

In `gui.py`, `_bind_keys` (`gui.py:1298-1299`), change:

```python
        @v.bind_key("b", overwrite=True)
        def _box(_v): self.activate_box_draw()
```

to:

```python
        @v.bind_key("b", overwrite=True)
        def _box(_v): self.activate_box_draw()

        @v.bind_key("l", overwrite=True)
        def _lasso_key(_v): self.activate_lasso_draw()
```

- [ ] **Step 7: Update the module docstring**

In `gui.py`, line 7, change:

```python
drawn bounding box), paint an anchor mask, re-run the image phase, and resume
```

to:

```python
drawn bounding box), paint an anchor mask (brush, or a freehand lasso that unions
a new area in), re-run the image phase, and resume
```

- [ ] **Step 8: Run the full suite and lint**

Run: `py -3 -m pytest -q`
Expected: all tests pass, same count as the end of Task 1 (this task adds no new automated tests, see
"Manual smoke test" below for why).

Run: `ruff check gui.py`
Expected: `All checks passed!`

- [ ] **Step 9: Manual smoke test in the live GUI**

This cannot be automated: it exercises napari's real mouse-drag event stream, which needs a live
display. Run against a real chain, e.g.:

```bash
py -3 gui.py --output-root "F:\ZhenLab\Data\output_masks\manual_verify_AIAL_AIAR" --neuron AIAL --chain 0 --anchor-only
```

Checklist:
- [ ] Press `l`. Console prints `[gui] lasso mode: drag a loop over the area to add, release to commit`.
- [ ] Click-drag a loop that starts on the visible mask, bulges out over background, and ends back on
  the mask. On release, the mask visibly extends to cover the new area; the drawn loop itself
  disappears (consumed, not left on screen).
- [ ] Press `Ctrl+Z`. The extension reverts; the mask returns to its pre-stroke shape.
- [ ] Press `l` again, click once without dragging (a degenerate stroke). The mask is unchanged, no
  error printed.
- [ ] Draw two lasso strokes in a row without pressing `l` again between them (mode should stay armed
  since nothing deactivates it). Both strokes commit; the second stroke's undo (`Ctrl+Z`) reverts only
  the second stroke, not the first.

- [ ] **Step 10: Add a CHANGELOG entry**

Read `docs/CHANGELOG.md`'s `## Contents` list and the most recent dated section (near the top, newest
first) to match the existing format, then add a new entry above it, dated with today's actual date,
titled along the lines of "lasso-add mask tool: a freehand loop unions into the current frame's mask,
undo-compatible with the paint brush." Cover: why (long boundary-stretch corrections were tedious with
brush-only painting; addition-only, since subtraction already has the eraser), what was built (the
`l` key, `self._lasso`, `_rasterize_lasso_fill`, `_on_lasso_drawn`, `data_setitem` for undo
compatibility), and the real napari-source-level facts this design leaned on (native
`add_polygon_lasso` mode, `data_setitem`'s undo tracking, the `ActionType.ADDED` filter that avoids a
re-entrancy guard). Run the `humanizer` skill on the new entry before treating it as final (project
convention, `CLAUDE.md`), and confirm zero em/en dashes in the file same as every other change this
session.

- [ ] **Step 11: Commit**

```bash
git add gui.py docs/CHANGELOG.md
git commit -m "Wire a freehand lasso-add tool into the mask review GUI"
```
