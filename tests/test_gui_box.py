"""Unit tests for the GUI box-prompt geometry helpers (gui._rect_to_xyxy /
_xyxy_to_rect / _box_on_frame), the pure conversion between an xyxy box (the
pipeline.Prompts.box_sam format) and a napari Shapes rectangle's (t, y, x) vertices.

Torch-free / napari-free: gui.py imports torch (via pipeline) and napari only lazily,
so importing the module and exercising these module-level helpers needs no GPU and no
viewer (same tactic as test_anchor_select).

Run either way:
    py -3 -m pytest tests/test_gui_box.py
    py -3 tests/test_gui_box.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import gui


# ---------------------------------------------------------------------------
# _rect_to_xyxy / _xyxy_to_rect: round-trip + corner-order independence
# ---------------------------------------------------------------------------

def test_xyxy_rect_round_trip():
    xyxy = np.array([10.0, 20.0, 110.0, 70.0])      # x0, y0, x1, y1
    rect = gui._xyxy_to_rect(xyxy, t=5)
    assert rect.shape == (4, 3)
    assert np.all(rect[:, 0] == 5)                  # every vertex on frame 5
    assert np.allclose(gui._rect_to_xyxy(rect), xyxy)


def test_rect_to_xyxy_is_corner_order_independent():
    # vertices given bottom-right-first; min/max still recover the same box
    verts = np.array([[3, 70, 110], [3, 70, 10], [3, 20, 10], [3, 20, 110]], float)
    assert np.allclose(gui._rect_to_xyxy(verts), [10, 20, 110, 70])


def test_rect_to_xyxy_handles_2d_vertices():
    # the recrop-picker layer is 2D (y, x); the last two columns are still (y, x)
    verts = np.array([[20, 10], [20, 110], [70, 110], [70, 10]], float)
    assert np.allclose(gui._rect_to_xyxy(verts), [10, 20, 110, 70])


# ---------------------------------------------------------------------------
# _box_on_frame: frame filtering, last-wins, empty
# ---------------------------------------------------------------------------

def test_box_on_frame_filters_by_frame():
    shapes = [
        gui._xyxy_to_rect([0, 0, 5, 5], t=2),
        gui._xyxy_to_rect([10, 10, 20, 20], t=7),
    ]
    assert np.allclose(gui._box_on_frame(shapes, 7), [10, 10, 20, 20])
    assert np.allclose(gui._box_on_frame(shapes, 2), [0, 0, 5, 5])


def test_box_on_frame_last_wins_on_same_frame():
    shapes = [
        gui._xyxy_to_rect([0, 0, 5, 5], t=4),
        gui._xyxy_to_rect([1, 1, 9, 9], t=4),       # redraw on the same frame
    ]
    assert np.allclose(gui._box_on_frame(shapes, 4), [1, 1, 9, 9])


def test_box_on_frame_none_when_absent_or_empty():
    shapes = [gui._xyxy_to_rect([0, 0, 5, 5], t=2)]
    assert gui._box_on_frame(shapes, 9) is None     # no box on frame 9
    assert gui._box_on_frame([], 0) is None          # no shapes at all


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
