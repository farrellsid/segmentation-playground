import numpy as np
from sam2_utils import membrane as mb


def test_membrane_map_responds_on_dark_ridge():
    patch = np.full((24, 24), 200, dtype=np.uint8)
    patch[:, 11:13] = 20  # a dark vertical ridge (membranes are dark)
    m = mb.membrane_map(patch)
    assert m.shape == (24, 24)
    assert m.dtype == np.float32
    assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0
    assert m[:, 10:14].mean() > m[:, 0:3].mean()  # ridge lights up vs flat area


def _rect_mask(h=30, w=30, y0=5, y1=25, x0=5, x1=25):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_spanning_membrane_flags_ridge_across_mask():
    mask = _rect_mask()
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[:, 14:16] = 1.0  # a ridge cutting the mask border-to-border
    spanning, frac = mb.spanning_membrane(mask, mem)
    assert spanning is True
    assert 0.0 < frac <= 0.5


def test_spanning_membrane_ignores_nucleus_loop():
    mask = _rect_mask()
    mem = np.zeros((30, 30), dtype=np.float32)
    # a closed loop well inside the mask (a nucleus), touching no mask border
    mem[10:20, 10] = 1.0; mem[10:20, 19] = 1.0
    mem[10, 10:20] = 1.0; mem[19, 10:20] = 1.0
    spanning, frac = mb.spanning_membrane(mask, mem)
    assert spanning is False
    assert frac == 0.0


def test_spanning_membrane_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    assert mb.spanning_membrane(mask, np.zeros((30, 30), np.float32)) == (False, 0.0)


def test_spanning_membrane_mask_fully_covered_by_membrane():
    mask = _rect_mask()
    mem = np.ones((30, 30), dtype=np.float32)  # membrane covers the whole mask
    spanning, frac = mb.spanning_membrane(mask, mem)
    assert spanning is False
    assert frac == 0.0


def test_boundary_on_membrane_high_when_edge_on_ridge():
    mask = _rect_mask()  # perimeter at rows/cols 5 and 24
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[4:26, 4:26] = 0.0
    mem[5, 5:25] = 1.0; mem[24, 5:25] = 1.0     # ridge along top+bottom edges
    mem[5:25, 5] = 1.0; mem[5:25, 24] = 1.0     # ridge along left+right edges
    on = mb.boundary_on_membrane(mask, mem)
    assert on > 0.8
    assert mb.boundary_on_membrane(mask, np.zeros((30, 30), np.float32)) == 0.0


def test_boundary_on_membrane_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    mem = np.ones((30, 30), dtype=np.float32)  # membrane everywhere, but no perimeter to test
    assert mb.boundary_on_membrane(mask, mem) == 0.0


def test_underfill_high_when_mask_inset_from_membrane_box():
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[5, 5:25] = 1.0; mem[24, 5:25] = 1.0     # a membrane box the cell lives in
    mem[5:25, 5] = 1.0; mem[5:25, 24] = 1.0
    inset = _rect_mask(y0=10, y1=15, x0=10, x1=15)   # small, room to grow
    filled = _rect_mask(y0=6, y1=24, x0=6, x1=24)    # fills the box
    assert mb.underfill_fraction(inset, mem, k=10) > 0.5
    assert mb.underfill_fraction(filled, mem, k=10) < 0.2


def test_underfill_empty_mask():
    assert mb.underfill_fraction(np.zeros((10, 10), bool), np.zeros((10, 10), np.float32)) == 0.0
