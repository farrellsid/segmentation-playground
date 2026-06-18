"""Unit tests for eval.erl, skeleton-based Expected Run Length.

Pure, on synthetic skeletons with hand-computed answers (like test_eval_metrics).
ERL = Σ run_len² / total_length, with merge labels contributing zero. Run:
    py -3 -m pytest tests/test_eval_erl.py
    py -3 tests/test_eval_erl.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from eval import erl as E


def line(n, neuron="A", prefix="a", length=1.0):
    """A straight chain of ``n`` nodes: edges (i,i-1,length). Returns
    (edges, node_neuron) with ids ``f'{prefix}{i}'``."""
    ids = [f"{prefix}{i}" for i in range(n)]
    edges = [(ids[i], ids[i - 1], length) for i in range(1, n)]
    neu = {nid: neuron for nid in ids}
    return ids, edges, neu


# =============================================================================
# core ERL
# =============================================================================

def test_perfect_single_label_erl_equals_total_length():
    ids, edges, neu = line(5)                       # total length 4
    labels = {nid: 7 for nid in ids}
    r = E.expected_run_length(edges, labels, neu)
    assert r["total_length"] == 4.0
    assert r["erl"] == 4.0                            # 4² / 4
    assert r["n_runs"] == 1 and r["max_run"] == 4.0
    assert r["n_merge_labels"] == 0
    assert r["n_split_edges"] == 0 and r["n_bg_edges"] == 0


def test_single_split_halves_runs():
    ids, edges, neu = line(5)                        # nodes a0..a4, total 4
    labels = {"a0": 7, "a1": 7, "a2": 7, "a3": 8, "a4": 8}
    r = E.expected_run_length(edges, labels, neu)
    # runs: {a0,a1,a2} len 2, {a3,a4} len 1; the a2-a3 edge is a split (len 0 run)
    assert math.isclose(r["erl"], (2 * 2 + 1 * 1) / 4)   # 5/4 = 1.25
    assert r["n_split_edges"] == 1
    assert r["n_runs"] == 2 and r["max_run"] == 2.0


def test_merge_zeroes_length():
    _, ea, na = line(3, neuron="A", prefix="a")      # total 2
    _, eb, nb = line(3, neuron="B", prefix="b")      # total 2
    edges = ea + eb
    neu = {**na, **nb}
    labels = {nid: 7 for nid in neu}                 # one label across both neurons
    r = E.expected_run_length(edges, labels, neu)
    assert r["total_length"] == 4.0
    assert r["erl"] == 0.0                            # everything is a merge
    assert r["n_merge_labels"] == 1
    assert r["merge_detail"] == {7: ["A", "B"]}
    assert r["n_merge_edges"] == 4 and r["n_runs"] == 0


def test_background_breaks_runs_but_stays_in_denominator():
    ids, edges, neu = line(4)                        # total 3
    labels = {"a0": 7, "a1": 7, "a2": 0, "a3": 7}    # a2 background (0)
    r = E.expected_run_length(edges, labels, neu)
    # only run {a0,a1} len 1; the two edges touching a2 are background
    assert math.isclose(r["erl"], 1.0 / 3.0)
    assert r["n_bg_edges"] == 2
    assert r["total_length"] == 3.0


def test_label_in_one_neuron_is_not_a_merge():
    _, ea, na = line(3, neuron="A", prefix="a")
    _, eb, nb = line(3, neuron="B", prefix="b")
    edges, neu = ea + eb, {**na, **nb}
    labels = {**{n: 1 for n in na}, **{n: 2 for n in nb}}  # distinct labels
    r = E.expected_run_length(edges, labels, neu)
    assert r["n_merge_labels"] == 0
    # two perfect runs of length 2 each: ERL = (4+4)/4 = 2
    assert r["erl"] == 2.0


# =============================================================================
# per-neuron breakdown
# =============================================================================

def test_per_neuron_erl():
    _, ea, na = line(3, neuron="A", prefix="a")      # total 2, perfect
    _, eb, nb = line(3, neuron="B", prefix="b")      # total 2, one split
    edges, neu = ea + eb, {**na, **nb}
    labels = {"a0": 1, "a1": 1, "a2": 1, "b0": 2, "b1": 2, "b2": 3}
    out = E.per_neuron_erl(edges, labels, neu)
    assert set(out) == {"A", "B"}
    assert out["A"]["erl"] == 2.0                    # 2² / 2
    assert math.isclose(out["B"]["erl"], 0.5)        # run len 1 -> 1²/2
    assert out["B"]["n_split_edges"] == 1


def test_per_neuron_merge_zeroes_both():
    _, ea, na = line(3, neuron="A", prefix="a")
    _, eb, nb = line(3, neuron="B", prefix="b")
    edges, neu = ea + eb, {**na, **nb}
    labels = {nid: 5 for nid in neu}                 # global merge
    out = E.per_neuron_erl(edges, labels, neu)
    assert out["A"]["erl"] == 0.0 and out["B"]["erl"] == 0.0
    assert out["A"]["n_merge_edges"] == 2


# =============================================================================
# merge tolerance (0.4): a few stray cross-neuron nodes shouldn't zero a neuron
# =============================================================================

def _stray_case():
    """Neuron A: 6-node line (len 5), all label 7. Neuron B: 3-node line (len 2),
    one stray node b0 also carries 7 (drifted onto A), b1/b2 carry 8."""
    _, ea, na = line(6, neuron="A", prefix="a")        # a0..a5, total 5
    _, eb, nb = line(3, neuron="B", prefix="b")        # b0..b2, total 2
    edges, neu = ea + eb, {**na, **nb}
    labels = {**{n: 7 for n in na}, "b0": 7, "b1": 8, "b2": 8}
    return edges, labels, neu


def test_strict_default_one_stray_node_zeroes_neuron():
    edges, labels, neu = _stray_case()
    r = E.expected_run_length(edges, labels, neu)       # defaults = strict
    assert r["n_merge_labels"] == 1                     # 7 spans A and B
    assert r["merge_detail"] == {7: ["A", "B"]}
    # only surviving run is {b1,b2} len 1; A is all merge-edges
    assert math.isclose(r["erl"], 1.0 / 7.0)


def test_count_tolerance_rescues_majority_label():
    edges, labels, neu = _stray_case()
    r = E.expected_run_length(edges, labels, neu, min_support_count=2)
    assert r["n_merge_labels"] == 0                     # B's single node 7 doesn't count
    # A run len 5, B run len 1 (b0-b1 is now a split, not a merge edge)
    assert math.isclose(r["erl"], (25.0 + 1.0) / 7.0)
    assert r["n_split_edges"] == 1 and r["n_merge_edges"] == 0


def test_frac_tolerance_rescues_majority_label():
    edges, labels, neu = _stray_case()                  # 7 has 7 nodes: A=6, B=1
    r = E.expected_run_length(edges, labels, neu, min_support_frac=0.5)
    assert r["n_merge_labels"] == 0                     # B holds 1/7 < 0.5
    assert math.isclose(r["erl"], 26.0 / 7.0)


def test_tolerance_still_flags_a_genuine_balanced_merge():
    _, ea, na = line(4, neuron="A", prefix="a")         # a0..a3
    _, eb, nb = line(4, neuron="B", prefix="b")         # b0..b3
    edges, neu = ea + eb, {**na, **nb}
    labels = {nid: 7 for nid in neu}                    # label 7: 4 of A, 4 of B
    r = E.expected_run_length(edges, labels, neu, min_support_frac=0.5)
    assert r["n_merge_labels"] == 1                     # each neuron holds 1/2 >= 0.5
    assert r["erl"] == 0.0


def test_per_neuron_tolerance_threads_through():
    edges, labels, neu = _stray_case()
    out = E.per_neuron_erl(edges, labels, neu, min_support_count=2)
    assert math.isclose(out["A"]["erl"], 25.0 / 5.0)    # full run 5 -> 5²/5 = 5
    assert math.isclose(out["B"]["erl"], 1.0 / 2.0)     # run 1 -> 1²/2; b0-b1 split


# =============================================================================
# loader + label sampling (the drive-dependent wire-in, tested synthetically)
# =============================================================================

def test_load_skeletons_lengths_use_resolution(tmp_path):
    csv = tmp_path / "agg.csv"
    # 3-node chain: n0 root, n1 +1 in x, n2 +1 in z.  res (2,2,50).
    csv.write_text(
        "node_id,parent_id,x,y,z,cell_name,is_vnode\n"
        "n0,,0,0,0,RMDVR!,False\n"
        "n1,n0,1,0,0,RMDVR!,False\n"
        "n2,n1,1,0,1,RMDVR!,False\n",
        encoding="utf-8",
    )
    sk = E.load_skeletons(csv, resolution_nm=(2, 2, 50))
    lengths = sorted(e[2] for e in sk.edges)
    assert lengths == [2.0, 50.0]                    # dx*2 and dz*50
    assert sk.neuron["n0"] == "RMDVR"                # normalize strips '!'
    assert math.isclose(sk.total_length_nm, 52.0)


def test_sample_node_labels_identity_and_oob():
    # two nodes on slice 3: one inside a labeled block, one out of bounds
    sk = E.Skeletons(
        edges=[("a", "b", 1.0)],
        xyz={"a": (2.0, 1.0, 3.0), "b": (99.0, 99.0, 3.0)},
        neuron={"a": "A", "b": "A"},
    )
    arr = np.zeros((5, 5), dtype=np.uint16)
    arr[1, 2] = 42                                   # row=y=1, col=x=2

    def label_slice_fn(z):
        assert z == 3
        return arr

    labels = E.sample_node_labels(sk, label_slice_fn)   # identity transform
    assert labels["a"] == 42
    assert labels["b"] == 0                          # out of bounds -> background


def test_sample_node_labels_missing_slice_is_background():
    sk = E.Skeletons(edges=[], xyz={"a": (0.0, 0.0, 9.0)}, neuron={"a": "A"})

    def label_slice_fn(z):
        raise KeyError(z)                            # slice not available

    labels = E.sample_node_labels(sk, label_slice_fn)
    assert labels["a"] == 0


# =============================================================================
# smoke test on the real local skeletons (no E: drive needed)
# =============================================================================

def test_real_skeletons_load_smoke():
    csv = (pathlib.Path(__file__).resolve().parent.parent
           / "data" / "groundtruth" / "skeletons_p280" / "aggregate_data_pv.csv")
    if not csv.exists():                             # skip if the pull isn't present
        import pytest
        pytest.skip("skeletons_p280 not present")
    import pandas as pd
    sk = E.load_skeletons(csv)
    n_nodes = len(pd.read_csv(csv, dtype={"node_id": str}))
    # nearly every non-root node must yield an edge: the float-vs-int parent-id
    # join bug dropped ~110k edges and shattered the trees, so guard the count.
    assert len(sk.edges) > 0.99 * (n_nodes - len(sk.neurons))
    assert sk.total_length_nm > 0
    assert len(sk.neurons) > 100                      # hundreds of neurons

    # Perfect per-neuron labeling -> no merges, one run per connected skeleton
    # tree (a few hundred), NOT tens of thousands of fragments.
    perfect = {nid: neu for nid, neu in sk.neuron.items()}
    rp = E.expected_run_length(sk.edges, perfect, sk.neuron)
    assert rp["n_merge_labels"] == 0
    assert rp["n_runs"] < 1000                        # ~#skeletons, not ~#nodes
    assert rp["erl"] / 1000 > 50.0                    # ceiling is tens+ of µm

    # a degenerate "all one label" prediction is a global merge -> ERL 0
    labels = {nid: 1 for nid in sk.neuron}
    r = E.expected_run_length(sk.edges, labels, sk.neuron)
    assert r["erl"] == 0.0 and r["n_merge_labels"] == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
