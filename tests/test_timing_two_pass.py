"""_append_timing must record BOTH passes of a two-pass (tier-2) chain.

Before the fix, `_run_one_chain` reassigned `state` to the second pass, so the
first `_sam` pass's time was discarded and `t_total` understated a tier-2
chain's true wall-clock by a whole pass.
"""
from types import SimpleNamespace

import pandas as pd

import batch


def test_append_timing_two_pass_includes_first_pass(tmp_path):
    state = SimpleNamespace(
        phase_seconds={"select anchor": 1.0, "propagate": 2.0},  # second pass
        phase_subseconds={},
        n_frames=10,
        tier1_seconds=5.0,   # first pass total, attached by _run_one_chain on a re-run
    )
    batch._append_timing(tmp_path, "AVAL", 3, state, peak_vram=1.2)
    row = pd.read_csv(tmp_path / "_timing.csv").iloc[0]
    assert row["t_pass1"] == 5.0
    assert row["t_total"] == 8.0   # 5.0 first pass + 3.0 second-pass phases


def test_append_timing_single_pass_has_no_pass1(tmp_path):
    state = SimpleNamespace(
        phase_seconds={"select anchor": 1.5}, phase_subseconds={}, n_frames=4)
    batch._append_timing(tmp_path, "AVAL", 0, state, peak_vram=1.0)
    row = pd.read_csv(tmp_path / "_timing.csv").iloc[0]
    assert pd.isna(row["t_pass1"])   # no second pass -> no first-pass time
    assert row["t_total"] == 1.5
