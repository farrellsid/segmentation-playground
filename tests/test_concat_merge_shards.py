"""Tests for the sharded-eval split + concat logic (eval.concat_merge_shards).

The split has to be disjoint + total (no neuron dropped or double-counted across
shards), and the concat has to reproduce, from the round-tripped shard CSVs, the same
summary a single-CPU score_run would have computed in memory."""
import json

import cv2
import numpy as np
import pandas as pd

from eval import concat_merge_shards as cc
from eval import merge_metric as mm


def test_split_neurons_disjoint_and_total():
    neurons = [f"N{i}" for i in range(16)]
    for n_shards in (1, 3, 5, 16):
        chunks = cc.split_neurons(neurons, n_shards)
        assert len(chunks) == n_shards
        flat = [x for c in chunks for x in c]
        assert flat == neurons                      # order preserved, nothing lost/dup
        sizes = [len(c) for c in chunks]
        assert max(sizes) - min(sizes) <= 1          # balanced


def test_split_neurons_one_per_shard_matches_exp_layout():
    neurons = [f"N{i}" for i in range(16)]
    chunks = cc.split_neurons(neurons, 16)
    assert chunks == [[n] for n in neurons]          # the 16-neuron array layout


def test_split_neurons_rejects_zero():
    import pytest
    with pytest.raises(ValueError):
        cc.split_neurons(["A"], 0)


def _write_chain(root, neuron, masks):
    d = root / neuron / "chain_00"
    (d / "masks").mkdir(parents=True)
    for z, arr in masks.items():
        cv2.imwrite(str(d / "masks" / f"mask_{z:04d}.png"), (arr > 0).astype("uint8") * 255)
    (d / "state.json").write_text("{}")


def test_concat_reproduces_whole_tree_summary(tmp_path):
    # Two neurons, one bleeds onto the other's node on one frame. Score each into its
    # own shard CSV, then concat: the stitched summary must equal the whole-tree score.
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root, "AVAL", {1400: a, 1401: a})
    _write_chain(root, "AVAR", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({
        "node_id": ["own_l0", "own_l1", "own_r0", "foreign_on_l"],
        "cell_name": ["AVAL", "AVAL", "AVAR", "AVAR"],
        "z": [1400, 1401, 1400, 1400],
        "x_tif": [120.0, 120.0, 900.0, 112.0],
        "y_tif": [120.0, 120.0, 900.0, 112.0],
    })

    # whole-tree reference (single CPU)
    _per_ref, ref = mm.score_run(root, annotate_df=df, radius=0, membrane_source=None)

    # sharded: one neuron per shard, each to its own shard CSV
    for i, chunk in enumerate(cc.split_neurons(["AVAL", "AVAR"], 2)):
        mm.score_run(root, annotate_df=df, radius=0, membrane_source=None,
                     neurons=chunk, out_csv=root / f"_merge_metric.shard_{i}.csv")

    out, summ = cc.concat_tree(root)
    assert out == root / "_merge_metric.csv"
    for k in ("n_chains", "n_frames", "foreign_frame_rate", "dropout_rate",
              "total_foreign_nodes"):
        assert summ[k] == ref[k], k
    # the stitched per-frame file has every frame from both neurons
    stitched = pd.read_csv(root / "_merge_metric.csv")
    assert len(stitched) == ref["n_frames"] == 3
    assert set(stitched["neuron"]) == {"AVAL", "AVAR"}


def test_concat_missing_shard_is_skipped(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root, "AVAL", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({"node_id": ["own"], "cell_name": ["AVAL"],
                       "z": [1400], "x_tif": [120.0], "y_tif": [120.0]})
    mm.score_run(root, annotate_df=df, radius=0, membrane_source=None,
                 neurons=["AVAL"], out_csv=root / "_merge_metric.shard_0.csv")
    # shard_1 was never produced (its task failed); concat still stitches shard_0
    _out, summ = cc.concat_tree(root)
    assert summ["n_frames"] == 1 and summ["n_chains"] == 1


def _write_merge_shard(root, i, rows):
    pd.DataFrame(rows).to_csv(root / f"_merge_metric.shard_{i}.csv", index=False)


def _write_z_shard(root, i, rows):
    pd.DataFrame(rows).to_csv(root / f"_z_consistency.shard_{i}.csv", index=False)


def test_concat_tree_stitches_z_consistency_shards(tmp_path, capsys):
    # Two shards, each with its own z-consistency shard, one row of which is a
    # dropout transition (iou/centroid_drift_px empty) to exercise the CSV
    # round-trip coercion in summarize_concat_z. This is the Finding 1 regression
    # test: score_run used to write every array task's z-consistency CSV to the
    # SAME unsharded path, so concat_tree needs its own shard-aware stitch to match.
    root = tmp_path / "run_merged"
    root.mkdir()
    _write_merge_shard(root, 0, [
        {"z": 1400, "neuron": "AVAL", "chain_idx": 0, "own_contained": True,
         "n_foreign": 0, "foreign_ids": "", "empty": False, "spanning_merge": None,
         "bled_fraction": None, "boundary_on_membrane": None, "underfill_fraction": None},
    ])
    _write_merge_shard(root, 1, [
        {"z": 1400, "neuron": "AVAR", "chain_idx": 0, "own_contained": True,
         "n_foreign": 0, "foreign_ids": "", "empty": False, "spanning_merge": None,
         "bled_fraction": None, "boundary_on_membrane": None, "underfill_fraction": None},
    ])
    _write_z_shard(root, 0, [
        {"z_from": 1400, "z_to": 1401, "gap": 1, "iou": 0.9,
         "centroid_drift_px": 1.0, "neuron": "AVAL", "chain_idx": 0},
        {"z_from": 1401, "z_to": 1402, "gap": 1, "iou": None,
         "centroid_drift_px": None, "neuron": "AVAL", "chain_idx": 0},  # dropout
    ])
    _write_z_shard(root, 1, [
        {"z_from": 1400, "z_to": 1401, "gap": 1, "iou": 0.3,
         "centroid_drift_px": 5.0, "neuron": "AVAR", "chain_idx": 0},
    ])

    _out, summ = cc.concat_tree(root)
    printed = capsys.readouterr().out

    assert (root / "_z_consistency.csv").exists()
    stitched = pd.read_csv(root / "_z_consistency.csv")
    assert len(stitched) == 3

    assert summ["n_transitions"] == 3
    assert summ["n_dropout_transitions"] == 1
    assert abs(summ["mean_z2z_iou"] - (0.9 + 0.3) / 2) < 1e-9
    assert abs(summ["mean_centroid_drift_px"] - (1.0 + 5.0) / 2) < 1e-9
    assert "mean_z2z_iou=" in printed


def test_concat_tree_no_z_shards_omits_z_segment(tmp_path, capsys):
    # No _z_consistency.shard_*.csv at all (a tree scored before this feature
    # existed): concat_tree must not crash, and the summary/printed line simply
    # omit the z segment, same as a tree with no z data.
    root = tmp_path / "run_merged"
    root.mkdir()
    _write_merge_shard(root, 0, [
        {"z": 1400, "neuron": "AVAL", "chain_idx": 0, "own_contained": True,
         "n_foreign": 0, "foreign_ids": "", "empty": False, "spanning_merge": None,
         "bled_fraction": None, "boundary_on_membrane": None, "underfill_fraction": None},
    ])
    _out, summ = cc.concat_tree(root)
    printed = capsys.readouterr().out
    assert "mean_z2z_iou" not in summ
    assert "mean_z2z_iou=" not in printed
    assert not (root / "_z_consistency.csv").exists()
