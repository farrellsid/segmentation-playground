"""Unit tests for sam2_utils.labels.LabelStore, the per-frame label engine.

Torch-free / napari-free, like test_alignment / test_anchor_select: labels.py is a
pure pandas CSV ledger, so the schema, idempotent upsert, qc-row feature copy,
anchor-verdict block, and the uniform un-flagged sample (the selection-bias
guard) all exercise on any box with a temp dir.

Run either way:
    py -3 -m pytest tests/test_labels.py
    py -3 tests/test_labels.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sam2_utils import labels as L


def _store():
    """A LabelStore on a fresh temp output root."""
    d = tempfile.mkdtemp()
    return L.LabelStore(d), pathlib.Path(d)


def _qc_df():
    """A small z-indexed qc.csv-shaped frame (columns match qc.compute_metrics)."""
    rows = []
    for z in range(1500, 1510):
        flag_count = 2 if z in (1503, 1504) else (1 if z == 1502 else 0)
        rows.append(dict(
            z=z, area=500 + z, n_components=1,
            skeleton_contained=(z not in (1503, 1504)),
            area_ratio=1.0, temporal_iou=0.9, pred_iou=0.8, logit_conf=0.95,
            flag_count=flag_count, flag=flag_count >= 1, intervene=flag_count >= 2,
        ))
    return pd.DataFrame(rows).set_index("z")


# ---------------------------------------------------------------------------
# schema + basic record
# ---------------------------------------------------------------------------

def test_record_creates_file_with_full_schema():
    store, root = _store()
    store.record("AIAL", 0, 1503, verdict="wrong", error_type="bleed",
                 role="flagged", source="reject", reviewer="sf")
    df = store.load()
    assert list(df.columns) == L.LABEL_COLS
    assert len(df) == 1
    r = df.iloc[0]
    assert r["neuron"] == "AIAL" and int(r["chain_idx"]) == 0 and int(r["z"]) == 1503
    assert r["verdict"] == "wrong" and r["error_type"] == "bleed" and r["role"] == "flagged"


def test_record_rejects_bad_verdict_role_errortype():
    store, _ = _store()
    for kw in (dict(verdict="maybe"),
               dict(verdict="ok", role="bogus"),
               dict(verdict="wrong", error_type="explode")):
        try:
            store.record("N", 0, 1, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kw}")


def test_qc_row_features_copied_and_rule_flagged_derived():
    store, _ = _store()
    qc = _qc_df()
    store.record("AIAL", 0, 1503, verdict="wrong", role="flagged",
                 qc_row=qc.loc[1503])
    r = store.load().iloc[0]
    assert int(r["area"]) == 500 + 1503
    assert int(r["flag_count"]) == 2
    assert bool(r["rule_flagged"]) is True            # derived from flag_count >= 1
    assert float(r["pred_iou"]) == 0.8


def test_rule_flagged_false_for_unflagged_qc_row():
    store, _ = _store()
    qc = _qc_df()
    store.record("AIAL", 0, 1500, verdict="ok", role="sampled", qc_row=qc.loc[1500])
    r = store.load().iloc[0]
    assert bool(r["rule_flagged"]) is False           # flag_count 0


def test_anchor_block_copied():
    store, _ = _store()
    anchor = {"passed": False, "reasons": ["frag", "noskel"], "contained": False,
              "largest_cc_frac": 0.4, "area_frac": 0.01}
    store.record("AIAL", 0, 1572, verdict="ok", role="anchor", anchor=anchor)
    r = store.load().iloc[0]
    assert bool(r["anchor_passed"]) is False
    assert r["anchor_reasons"] == "frag,noskel"
    assert bool(r["anchor_contained"]) is False


def test_anchor_contained_none_is_blank_not_false():
    store, _ = _store()
    store.record("N", 0, 1, verdict="ok", anchor={"passed": True, "reasons": [],
                                                   "contained": None})
    r = store.load().iloc[0]
    # abstain (no positive point) must read as blank, not a hard False
    assert r["anchor_contained"] in ("", None) or pd.isna(r["anchor_contained"])


# ---------------------------------------------------------------------------
# idempotent upsert
# ---------------------------------------------------------------------------

def test_record_is_idempotent_per_key():
    store, _ = _store()
    store.record("AIAL", 0, 1503, verdict="ok", role="flagged")
    store.record("AIAL", 0, 1503, verdict="wrong", role="flagged", error_type="under")
    df = store.load()
    assert len(df) == 1                               # overwritten, not duplicated
    assert df.iloc[0]["verdict"] == "wrong" and df.iloc[0]["error_type"] == "under"


def test_distinct_keys_accumulate():
    store, _ = _store()
    store.record("AIAL", 0, 1503, verdict="wrong", role="flagged")
    store.record("AIAL", 0, 1504, verdict="wrong", role="flagged")
    store.record("AIAL", 1, 1503, verdict="ok", role="flagged")   # different chain
    assert len(store.load()) == 3


# ---------------------------------------------------------------------------
# uniform un-flagged sample (the silent-error window)
# ---------------------------------------------------------------------------

def test_sample_unflagged_picks_only_unflagged_and_is_reproducible():
    store, _ = _store()
    qc = _qc_df()
    out = store.sample_unflagged("AIAL", 0, qc, n=3, seed=0, reviewer="sf")
    zs = sorted(r["z"] for r in out)
    # flagged z (1502 flag=1, 1503/1504 flag>=1) must never be sampled
    assert all(z not in (1502, 1503, 1504) for z in zs)
    df = store.load()
    assert (df["role"] == "sampled").all() and (df["verdict"] == "ok").all()
    # same seed -> same draw
    store2, _ = _store()
    out2 = store2.sample_unflagged("AIAL", 0, qc, n=3, seed=0)
    assert sorted(r["z"] for r in out2) == zs


def test_sample_unflagged_excludes_requested_z():
    store, _ = _store()
    qc = _qc_df()
    # exclude all but one unflagged z -> only that one can be sampled
    unflagged = [1500, 1501, 1505, 1506, 1507, 1508, 1509]
    out = store.sample_unflagged("AIAL", 0, qc, n=10, exclude_z=unflagged[1:])
    assert [r["z"] for r in out] == [1500]


def test_sample_unflagged_handles_z_column_or_index():
    store, _ = _store()
    qc = _qc_df().reset_index()                       # z as a column, not the index
    out = store.sample_unflagged("AIAL", 0, qc, n=2, seed=1)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_counts():
    store, _ = _store()
    store.record("AIAL", 0, 1503, verdict="wrong", role="flagged")
    store.record("AIAL", 0, 1572, verdict="ok", role="anchor")
    store.record("AIAL", 1, 1503, verdict="ok", role="sampled")
    s = store.summary()
    assert s["n"] == 3 and s["n_chains"] == 2
    assert s["by_role"]["flagged"] == 1 and s["by_verdict"]["ok"] == 2


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
