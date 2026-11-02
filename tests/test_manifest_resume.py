"""Regression test for a real crash found on a Narval cluster run: resuming a
manifest where every row's anchor_reasons was blank (the common case, an
empty anchor_reasons means the anchor passed) reloads that column as float64
(pandas infers an all-blank CSV column as NaN), so the next _update_row call
raises pandas.errors.LossySetitemError trying to write a real empty string
into it. load_or_init_manifest already guarded status/error against this
exact failure mode; anchor_reasons was missing from that guard."""
from pathlib import Path

import pandas as pd

from batch import MANIFEST_COLUMNS, _update_row, load_or_init_manifest


def test_resume_with_blank_anchor_reasons_does_not_raise(tmp_path):
    manifest_path = tmp_path / "_manifest.csv"
    all_chains = [("X", 0, {}), ("X", 1, {})]

    # First run: every chain finishes with a blank anchor_reasons (anchor
    # passed cleanly), matching what a real successful chain writes.
    manifest = load_or_init_manifest(manifest_path, all_chains)
    for neuron, idx, _ in all_chains:
        _update_row(manifest, neuron, idx, status="done", anchor_reasons="", error="")
    manifest.to_csv(manifest_path, index=False)

    # Resume with the SAME chain set (no new rows added). This is the real
    # Narval scenario, e.g. a Slurm requeue restarting the same array task:
    # load_or_init_manifest's pd.concat branch for newly-added chains never
    # runs, so nothing incidentally widens anchor_reasons back to object. An
    # earlier, wrong version of this test added a third chain here, which
    # made the concat path mask the bug it was meant to catch.
    resumed = load_or_init_manifest(manifest_path, all_chains)

    # A chain gets retried on resume (its earlier run died some other way,
    # e.g. OOM) and writes its fields again, including a real empty-string
    # anchor_reasons. This is the exact call that crashed on Narval:
    # pandas.errors.LossySetitemError if anchor_reasons reloaded as float64.
    _update_row(resumed, "X", 0, status="done", anchor_reasons="", error="")

    assert resumed.loc[resumed["chain_idx"] == 0, "anchor_reasons"].iloc[0] == ""
    assert set(resumed.columns) == set(MANIFEST_COLUMNS)
