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
