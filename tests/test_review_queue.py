"""Unit tests for sam2_utils.review_queue.ReviewQueue, the work queue + the
GUI-owned review-status ledger (_review.csv), kept separate from the batch's
execution manifest (_manifest.csv).

Torch-free / napari-free: review_queue is pure pandas over the on-disk CSVs, so
the queue definition, the disposition upsert, the crash-recovery (in_review
re-surfacing), and the manifest/review separation all test on a temp dir.

Run either way:
    py -3 -m pytest tests/test_review_queue.py
    py -3 tests/test_review_queue.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from sam2_utils import review_queue as RQ


def _root_with_manifest(rows):
    """A temp output root with a _manifest.csv built from (neuron, chain_idx, status)."""
    d = pathlib.Path(tempfile.mkdtemp())
    pd.DataFrame([{"neuron": n, "chain_idx": i, "status": s} for n, i, s in rows]).to_csv(
        d / "_manifest.csv", index=False)
    return d


def _basic_manifest():
    return _root_with_manifest([
        ("AIAL", 0, "flagged"),
        ("AIAL", 1, "done"),
        ("AIAL", 2, "done"),
        ("AIYL", 12, "flagged"),
        ("AIAR", 8, "failed"),
    ])


# ---------------------------------------------------------------------------
# flagged chains + pending queue
# ---------------------------------------------------------------------------

def test_flagged_chains_only_flagged():
    q = RQ.ReviewQueue(_basic_manifest())
    assert q.flagged_chains() == [("AIAL", 0), ("AIYL", 12)]


def test_pending_is_flagged_minus_disposed():
    root = _basic_manifest()
    q = RQ.ReviewQueue(root)
    assert q.pending() == [("AIAL", 0), ("AIYL", 12)]
    q.set_status("AIAL", 0, RQ.APPROVED, reviewer="sf")
    assert q.pending() == [("AIYL", 12)]              # approved drops out
    q.set_status("AIYL", 12, RQ.CORRECTED)
    assert q.pending() == []                          # all disposed


def test_rejected_also_drops_out():
    q = RQ.ReviewQueue(_basic_manifest())
    q.set_status("AIAL", 0, RQ.REJECTED)
    assert ("AIAL", 0) not in q.pending()


# ---------------------------------------------------------------------------
# in_review / crash recovery
# ---------------------------------------------------------------------------

def test_claim_sets_in_review_and_resurfaces_by_default():
    q = RQ.ReviewQueue(_basic_manifest())
    q.claim("AIAL", 0, reviewer="sf")
    assert q.status_of("AIAL", 0) == RQ.IN_REVIEW
    # default include_in_review=True: a crashed/abandoned session's chain re-surfaces
    assert ("AIAL", 0) in q.pending(include_in_review=True)
    # hide it when another reviewer is actively on it
    assert ("AIAL", 0) not in q.pending(include_in_review=False)


# ---------------------------------------------------------------------------
# review ledger separation + idempotency
# ---------------------------------------------------------------------------

def test_review_ledger_is_separate_file_and_does_not_touch_manifest():
    root = _basic_manifest()
    q = RQ.ReviewQueue(root)
    q.set_status("AIAL", 0, RQ.CORRECTED, reviewer="sf")
    assert (root / "_review.csv").exists()
    # manifest execution status untouched (still 'flagged')
    man = pd.read_csv(root / "_manifest.csv")
    row = man[(man.neuron == "AIAL") & (man.chain_idx == 0)]
    assert row["status"].iloc[0] == "flagged"


def test_set_status_idempotent_per_chain():
    root = _basic_manifest()
    q = RQ.ReviewQueue(root)
    q.set_status("AIAL", 0, RQ.IN_REVIEW)
    q.set_status("AIAL", 0, RQ.APPROVED)
    rev = pd.read_csv(root / "_review.csv")
    sub = rev[(rev.neuron == "AIAL") & (rev.chain_idx == 0)]
    assert len(sub) == 1 and sub["review_status"].iloc[0] == RQ.APPROVED


def test_set_status_rejects_unknown_status():
    q = RQ.ReviewQueue(_basic_manifest())
    try:
        q.set_status("AIAL", 0, "halfway")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown status")


def test_status_of_defaults_unreviewed():
    q = RQ.ReviewQueue(_basic_manifest())
    assert q.status_of("AIAL", 0) == RQ.UNREVIEWED


# ---------------------------------------------------------------------------
# triage_for + refresh + summary + empty/missing
# ---------------------------------------------------------------------------

def test_triage_for_filters_to_chain():
    root = _basic_manifest()
    pd.DataFrame([
        {"neuron": "AIAL", "chain_idx": 0, "z": 1548, "flag_count": 4, "reasons": "noskel"},
        {"neuron": "AIAL", "chain_idx": 0, "z": 1549, "flag_count": 4, "reasons": "area"},
        {"neuron": "AIYL", "chain_idx": 12, "z": 1487, "flag_count": 4, "reasons": "noskel"},
    ]).to_csv(root / "_triage.csv", index=False)
    q = RQ.ReviewQueue(root)
    t = q.triage_for("AIAL", 0)
    assert len(t) == 2 and set(t["z"]) == {1548, 1549}


def test_refresh_picks_up_new_flags():
    root = _basic_manifest()
    q = RQ.ReviewQueue(root)
    assert len(q.flagged_chains()) == 2
    # a still-running batch flags another chain
    man = pd.read_csv(root / "_manifest.csv")
    man.loc[man.chain_idx == 8, "status"] = "flagged"
    man.to_csv(root / "_manifest.csv", index=False)
    q.refresh()
    assert ("AIAR", 8) in q.flagged_chains()


def test_missing_manifest_is_empty_not_error():
    d = pathlib.Path(tempfile.mkdtemp())
    q = RQ.ReviewQueue(d)
    assert q.flagged_chains() == [] and q.pending() == []
    assert q.triage_for("X", 0).empty


def test_summary_shape():
    root = _basic_manifest()
    q = RQ.ReviewQueue(root)
    q.set_status("AIAL", 0, RQ.APPROVED)
    s = q.summary()
    assert s["n_flagged"] == 2 and s["n_pending"] == 1
    assert s["by_review_status"].get(RQ.APPROVED) == 1


# ---------------------------------------------------------------------------
# all_chains: the openable universe (on-disk chain dirs), the GUI 'everything' mode
# ---------------------------------------------------------------------------

def _mkchains(root, specs):
    """Create <neuron>/chain_NN directories under root from (neuron, idx) specs."""
    for neuron, idx in specs:
        (root / neuron / f"chain_{idx:02d}").mkdir(parents=True, exist_ok=True)
    return root


def test_all_chains_enumerates_on_disk_dirs_sorted():
    root = _basic_manifest()
    _mkchains(root, [("AIAL", 2), ("AIAL", 0), ("AIYL", 12), ("AIAR", 8)])
    q = RQ.ReviewQueue(root)
    # sorted by (neuron, chain_idx); a superset of flagged, includes done/failed
    assert q.all_chains() == [("AIAL", 0), ("AIAL", 2), ("AIAR", 8), ("AIYL", 12)]


def test_all_chains_ignores_top_level_files_and_non_chain_dirs():
    root = _basic_manifest()                       # writes _manifest.csv at top level
    _mkchains(root, [("AIAL", 0)])
    (root / "AIAL" / "notes").mkdir()              # a non-chain dir inside a neuron
    q = RQ.ReviewQueue(root)
    assert q.all_chains() == [("AIAL", 0)]          # _manifest.csv + 'notes' excluded


def test_all_chains_can_include_chains_absent_from_manifest():
    root = _basic_manifest()
    _mkchains(root, [("AIAL", 0), ("ZZZ", 5)])      # ZZZ has no manifest row
    q = RQ.ReviewQueue(root)
    assert ("ZZZ", 5) in q.all_chains()


def test_all_chains_missing_root_is_empty():
    d = pathlib.Path(tempfile.mkdtemp()) / "nope"   # never created
    assert RQ.ReviewQueue(d).all_chains() == []


# ---------------------------------------------------------------------------
# chain_status / manifest_status: the picker badge
# ---------------------------------------------------------------------------

def test_manifest_status_reads_execution_status_or_none():
    q = RQ.ReviewQueue(_basic_manifest())
    assert q.manifest_status("AIAL", 0) == "flagged"
    assert q.manifest_status("AIAL", 1) == "done"
    assert q.manifest_status("AIAR", 8) == "failed"
    assert q.manifest_status("ZZZ", 9) is None      # no manifest row


def test_chain_status_review_disposition_wins():
    q = RQ.ReviewQueue(_basic_manifest())
    q.set_status("AIAL", 0, RQ.CORRECTED)           # flagged in manifest, but corrected
    assert q.chain_status("AIAL", 0) == RQ.CORRECTED
    q.claim("AIYL", 12)                             # in_review beats manifest 'flagged'
    assert q.chain_status("AIYL", 12) == RQ.IN_REVIEW


def test_chain_status_falls_back_to_manifest_then_unreviewed():
    q = RQ.ReviewQueue(_basic_manifest())
    assert q.chain_status("AIAL", 1) == "done"      # no review row -> manifest status
    assert q.chain_status("ZZZ", 9) == RQ.UNREVIEWED  # neither -> unreviewed


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
