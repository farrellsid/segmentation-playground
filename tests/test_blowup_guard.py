import numpy as np
from pipeline.propagate import apply_blowup_guard


def _mask(n_true, shape=(50, 50)):
    m = np.zeros(shape, dtype=bool)
    m.flat[:n_true] = True
    return m


def _chain(areas, obj_id=1):
    vs = {i: {obj_id: _mask(a)} for i, a in enumerate(areas)}
    return vs, {i: 0.9 for i in range(len(areas))}, {i: 0.9 for i in range(len(areas))}


def test_guard_replaces_blowup_with_nearest_accepted():
    # Baseline kept below 100 px on purpose: with the default (50, 50) = 2500 px mask
    # budget, a baseline near 100-110 (as in the task brief's own example) makes
    # 25x-median already exceed the 2500 px ceiling, so no area could ever be flagged.
    # Frame 4 at 1500 px is ~29x the ~51 px median and safely under the pixel ceiling.
    vs, fc, pi = _chain([50, 55, 45, 50, 1500, 52])  # frame 4 is 1500 vs median ~51
    guarded = apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0)
    assert guarded == {4}
    assert int(vs[4][1].sum()) == int(vs[3][1].sum())     # replaced by nearest accepted (frame 3)
    assert fc[4] == 0.0 and pi[4] == 0.0                  # flagged for review
    assert fc[0] == 0.9                                   # others untouched


def test_guard_ignores_normal_variation():
    vs, fc, pi = _chain([100, 100, 100, 100, 900])        # 9x median, below 25x
    assert apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0) == set()
    assert int(vs[4][1].sum()) == 900                     # unchanged


def test_guard_noop_when_too_few_accepted():
    vs, fc, pi = _chain([100, 5000])                      # only 2 non-empty -> no baseline
    assert apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0, min_accepted=3) == set()
