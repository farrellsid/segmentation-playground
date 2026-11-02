"""Unit tests for the lasso-add mask tool: draw a freehand loop over the area to
extend, release, and its interior unions into the current frame's mask (see
docs/superpowers/specs/2026-08-12-lasso-add-mask-tool-design.md).

Two layers of test here:

1. gui._rasterize_lasso_fill, the pure rasterize-and-union step. Torch-free and
   napari-free: gui.py imports torch (via pipeline), napari, and cv2 all lazily, so
   importing the module and exercising this module-level helper needs no GPU, no
   display, and no viewer (same tactic as test_gui_box.py).
2. The GUI event wiring, _on_lasso_drawn, driven by replaying napari's real
   freehand-lasso draw sequence against a napari.components.ViewerModel plus real
   Shapes and Labels layers. That needs napari's data model but still no Qt widgets,
   no display, and no GPU. It exists because the first version of this feature
   crashed on every real stroke (a re-entrancy bug in the inline shape trim) and no
   test caught it: the plan had assumed this wiring could not be tested headlessly.

Run either way:
    py -3 -m pytest tests/test_gui_lasso.py
    py -3 tests/test_gui_lasso.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from copy import copy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import gui

# napari is a GUI-only dependency (see the "Two layers" note above), not in this
# project's test/dev extras, so CI does not install it. The layer-2 tests below
# replay napari's real Shapes/ViewerModel draw sequence to catch a reentrancy bug
# that only reproduces against real napari internals, so they need the real package,
# not a stub. Skipped per-test (not a module-level importorskip) so the layer-1
# tests above them, genuinely torch/napari-free, still run in CI.
_HAS_NAPARI = importlib.util.find_spec("napari") is not None
_needs_napari = pytest.mark.skipif(not _HAS_NAPARI, reason="napari not installed (CI)")


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
# _on_lasso_drawn: the real napari draw sequence, headless
# ---------------------------------------------------------------------------

class _StubReviewData:
    """Just the two ReviewData fields _on_lasso_drawn reads."""
    obj_id = 1
    anchor_idx = 0


def _build_headless_gui(shape=(3, 64, 64)):
    """A ViewerModel plus real Shapes/Labels layers, wired to a ReviewGUI stand-in.

    object.__new__ skips ReviewGUI.__init__ (which would want a real chain on disk,
    a Qt viewer, and torch); only the attributes _on_lasso_drawn and _clear_lasso
    actually touch are set. Everything napari-side is real: a real Shapes layer in
    add_polygon_lasso mode, a real Labels layer, a real dims slider.
    """
    from napari.components import ViewerModel

    viewer = ViewerModel()
    mask = viewer.add_labels(np.zeros(shape, dtype=np.uint16), name="mask")
    lasso = viewer.add_shapes(name="lasso", ndim=3, edge_color=gui._LASSO_EDGE_COLOR,
                              face_color="transparent", edge_width=1.0, opacity=0.8)

    g = object.__new__(gui.ReviewGUI)
    g.viewer = viewer
    g._mask = mask
    g._lasso = lasso
    g.data = _StubReviewData()

    lasso.events.data.connect(g._on_lasso_drawn)
    lasso.mode = "add_polygon_lasso"
    viewer.dims.current_step = (0, 0, 0)
    return g, lasso, mask


def _feed_vertex(layer, coord):
    """One drag-move vertex, exactly what napari's own add_vertex_to_path does
    (napari/layers/shapes/_shapes_mouse_bindings.py) minus the MouseEvent bits."""
    index = layer._moving_value[0]
    vertices = np.concatenate((layer._data_view.shapes[index].data, [coord]), axis=0)
    layer._value = (index, len(vertices) - 1)
    layer._moving_value = copy(layer._value)
    layer._data_view.edit(index, vertices)


def _replay_lasso_stroke(layer, frame=0, dx=0):
    """Replay a real freehand lasso: press (initiate_polygon_draw), a run of drag
    vertices, then the mouse-up finish (_finish_drawing). The vertices are spread
    well apart so napari's RDP simplification inside _finish_drawing keeps more than
    three of them, matching a genuine drag rather than a click."""
    from napari.layers.shapes._shapes_mouse_bindings import initiate_polygon_draw

    ring = ([(frame, 10.0, float(x) + dx) for x in range(10, 51, 5)]
            + [(frame, float(y), 50.0 + dx) for y in range(15, 51, 5)]
            + [(frame, 50.0, float(x) + dx) for x in range(45, 9, -5)]
            + [(frame, float(y), 10.0 + dx) for y in range(45, 9, -5)])

    initiate_polygon_draw(layer, ring[0])
    for coord in ring[1:]:
        _feed_vertex(layer, coord)
    layer._finish_drawing()


@_needs_napari
def test_real_lasso_stroke_does_not_crash_and_unions_the_mask():
    """Regression test for the re-entrancy crash found by the final whole-branch
    review. The original code cleared the consumed shape inline (self._lasso.data =
    shapes_data[:-1]) from inside the 'data' event that napari's own
    Shapes._finish_drawing had just emitted. The Shapes.data setter opens with an
    unconditional _finish_drawing() call, so that assignment re-entered the outer
    _finish_drawing while its index was already reset to None, and every real stroke
    died on TypeError: list indices must be integers or slices, not NoneType. The
    clear is now deferred with QTimer.singleShot(0, ...).

    Scope note: the assertions here are the synchronously verifiable ones (no
    exception, mask unioned correctly). The deferred clear itself only fires under a
    running Qt event loop, which this test deliberately does not start, so
    self._lasso ending up empty is checked in the separate opt-in test below."""
    g, lasso, mask = _build_headless_gui()

    _replay_lasso_stroke(lasso, frame=0)      # raises on the pre-fix code

    painted = int((mask.data[0] == 1).sum())
    assert painted > 0, "the lasso loop's interior should have been unioned in"
    assert not mask.data[1].any(), "only the current frame should have been touched"
    assert lasso._is_creating is False, "the layer must not be left wedged mid-draw"

    # a second consecutive stroke must also work (the pre-fix bug broke this one too)
    _replay_lasso_stroke(lasso, frame=0, dx=8)
    assert int((mask.data[0] == 1).sum()) > painted


@_needs_napari
def test_out_of_range_frame_is_ignored():
    """The dims slider can outrun the mask's real T extent (any layer with a longer
    T stretches it, including the lasso layer itself while a shape sits on it), so
    _on_lasso_drawn guards the index rather than letting it IndexError."""
    g, lasso, mask = _build_headless_gui()          # mask T extent is 3
    # a shape parked past the mask's T extent stretches the viewer's own T range,
    # which is what lets the slider move past frame 2 in the first place
    lasso.data = [np.array([[99.0, 10.0, 10.0], [99.0, 10.0, 20.0], [99.0, 20.0, 20.0]])]
    g.viewer.dims.current_step = (99, 0, 0)
    assert g._current_frame() == 99
    mask.data[:] = 0

    _replay_lasso_stroke(lasso, frame=99)           # IndexError without the guard

    assert not mask.data.any()


@_needs_napari
def test_deferred_clear_empties_the_lasso_layer():
    """The other half of the fix: once a Qt event loop turn happens, the deferred
    _clear_lasso really does drop the consumed shape. Needs a QApplication, so it
    quietly no-ops where one cannot be created (headless CI without a Qt platform
    plugin); the test above covers the crash itself with no Qt at all."""
    try:
        from qtpy.QtCore import QEventLoop, QTimer
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
    except Exception:                                   # noqa: BLE001 - optional dep
        return

    g, lasso, mask = _build_headless_gui()
    _replay_lasso_stroke(lasso, frame=0)
    assert len(lasso.data) == 1                         # not cleared yet, by design

    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()

    assert len(lasso.data) == 0
    assert int((mask.data[0] == 1).sum()) > 0
    del app


@_needs_napari
def test_guard_trip_still_clears_the_lasso_layer_out_of_range_frame():
    """A stroke that trips the frame_idx bounds guard must still end up cleared from
    the layer, not left stuck: a stuck loop is itself part of what stretches the
    viewer's dims T-extent past the mask's real extent, so an un-cleared loop is
    self-sustaining. _on_lasso_drawn now runs its guarded body in a try/finally so the
    deferred clear fires regardless of which guard returns. Needs a QApplication for
    the deferred clear to actually fire; quietly no-ops where one cannot be created,
    matching the tests above."""
    try:
        from qtpy.QtCore import QEventLoop, QTimer
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
    except Exception:                                   # noqa: BLE001 - optional dep
        return

    g, lasso, mask = _build_headless_gui()          # mask T extent is 3
    # stub _current_frame directly rather than parking an extra shape to stretch the
    # dims T-extent (test_out_of_range_frame_is_ignored's trick): parking a shape
    # first fires its own ADDED event through the already-connected handler, which
    # would confuse the shape-count assertions this test cares about
    g._current_frame = lambda: 99                       # forces the frame_idx guard to trip

    _replay_lasso_stroke(lasso, frame=0)
    assert len(lasso.data) == 1                         # not cleared yet, by design

    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()

    assert len(lasso.data) == 0, "a guard trip must not leave the loop stuck on the layer"
    assert not mask.data.any()
    del app


@_needs_napari
def test_guard_trip_still_clears_the_lasso_layer_no_mask():
    """Same fix, the other named guard: self._mask is None (e.g. between chains)."""
    try:
        from qtpy.QtCore import QEventLoop, QTimer
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
    except Exception:                                   # noqa: BLE001 - optional dep
        return

    g, lasso, mask = _build_headless_gui()
    g._mask = None

    _replay_lasso_stroke(lasso, frame=0)
    assert len(lasso.data) == 1                         # not cleared yet, by design

    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()

    assert len(lasso.data) == 0, "a guard trip must not leave the loop stuck on the layer"
    del app


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
