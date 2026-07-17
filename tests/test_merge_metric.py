import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
from eval import merge_metric as mm
from sam2_utils import membrane as mb

def test_nodes_by_z_groups_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"],
        "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401],
        "x_tif": [800.0, 1600.0, 240.0],
        "y_tif": [80.0, 160.0, 800.0],
    })
    got = mm.nodes_by_z(df, scale=8)
    assert set(got) == {1400, 1401}
    assert sorted(got[1400]) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert got[1401] == [(30.0, 100.0, "AVAL", "c")]


def test_containment_own_and_foreign():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True  # a blob at grid (10..19, 10..19)
    nodes = [
        (15.0, 15.0, "AVAL", "own"),    # inside, own neuron
        (14.0, 14.0, "AVAR", "foreign_in"),   # inside, foreign
        (40.0, 40.0, "AVAR", "foreign_out"),  # outside
    ]
    assert mm.own_contained(mask, 0, 0, (15.0, 15.0), radius=0) is True
    assert mm.own_contained(mask, 0, 0, (40.0, 40.0), radius=0) is False
    hits = mm.foreign_hits(mask, 0, 0, nodes, own_neuron="AVAL", radius=0)
    assert hits == ["foreign_in"]

def test_containment_respects_offset():
    mask = np.ones((10, 10), dtype=bool)  # a crop placed at (x0=100, y0=200)
    # a foreign node at grid (105, 205) is local (5, 5): inside
    hits = mm.foreign_hits(mask, 100, 200, [(105.0, 205.0, "X", "n")], own_neuron="AVAL", radius=0)
    assert hits == ["n"]
    # a foreign node at grid (5, 5) is local (-95, -195): outside
    assert mm.foreign_hits(mask, 100, 200, [(5.0, 5.0, "X", "n")], "AVAL", 0) == []


def _write_chain(tmp_path, name, masks):
    """masks: {z: 2D uint8 array}. Writes a legacy chain (no crop_window)."""
    d = tmp_path / name
    (d / "masks").mkdir(parents=True)
    for z, arr in masks.items():
        cv2.imwrite(str(d / "masks" / f"mask_{z:04d}.png"), (arr > 0).astype("uint8") * 255)
    (d / "state.json").write_text("{}")  # no crop_window -> legacy _sam, offset (0,0)
    return d


def test_score_chain_flags_foreign_and_dropout(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1   # z1400: covers own+foreign
    b = np.zeros((50, 50), dtype=np.uint8)                        # z1401: empty (dropout)
    d = _write_chain(tmp_path, "AVAL_chain00", {1400: a, 1401: b})
    nbz = {
        1400: [(15.0, 15.0, "AVAL", "own0"), (14.0, 14.0, "AVAR", "f0")],
        1401: [(15.0, 15.0, "AVAL", "own1")],
    }
    recs = {r["z"]: r for r in mm.score_chain(d, "AVAL", nbz, radius=0)}
    assert recs[1400]["own_contained"] and recs[1400]["n_foreign"] == 1
    assert recs[1400]["foreign_ids"] == ["f0"] and not recs[1400]["empty"]
    assert recs[1401]["empty"] and not recs[1401]["own_contained"]
    assert recs[1401]["n_foreign"] == 0


def test_score_run_aggregates(tmp_path, monkeypatch):
    # one neuron AVAL with a chain that bleeds onto AVAR's node on one of two frames
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    b = np.zeros((50, 50), dtype=np.uint8); b[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a, 1401: b})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({
        "node_id": ["own0", "own1", "f0"], "cell_name": ["AVAL", "AVAL", "AVAR"],
        "z": [1400, 1401, 1400], "x_tif": [120.0, 120.0, 112.0], "y_tif": [120.0, 120.0, 112.0],
    })
    per, summ = mm.score_run(root, annotate_df=df, radius=0)
    assert summ["n_chains"] == 1 and summ["n_frames"] == 2
    assert summ["total_foreign_nodes"] == 1        # f0 hit on z1400 only
    assert abs(summ["foreign_frame_rate"] - 0.5) < 1e-9
    assert summ["dropout_rate"] == 0.0
    assert (root / "_merge_metric.csv").exists()
    assert set(per["neuron"]) == {"AVAL"}


def test_format_summary_is_one_line():
    s = mm.format_summary("neg", {
        "n_chains": 100, "n_frames": 8052, "foreign_frame_rate": 0.031,
        "dropout_rate": 0.12, "total_foreign_nodes": 274})
    assert "neg" in s and "0.031" in s and "\n" not in s


def test_membrane_source_crops_and_maps(monkeypatch):
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    frame[:, 20:22] = 20  # a dark ridge in the full _sam frame
    monkeypatch.setattr(mm.pipeline, "load_frame_sam",
                        lambda z, *, scale, frame_store=None: (frame, (0, 0)))
    src = mm.MembraneSource(scale=8)
    m = src.map_for(1400, x0=10, y0=10, h=20, w=20)
    assert m is not None and m.shape == (20, 20)
    assert float(m.max()) <= 1.0


def test_membrane_source_missing_frame_returns_none(monkeypatch):
    def boom(z, *, scale, frame_store=None):
        raise FileNotFoundError(z)
    monkeypatch.setattr(mm.pipeline, "load_frame_sam", boom)
    src = mm.MembraneSource(scale=8)
    assert src.map_for(1400, 0, 0, 10, 10) is None


def test_membrane_source_failed_z_is_cached_not_retried(monkeypatch):
    calls = {"n": 0}
    def boom(z, *, scale, frame_store=None):
        calls["n"] += 1
        raise FileNotFoundError(z)
    monkeypatch.setattr(mm.pipeline, "load_frame_sam", boom)
    src = mm.MembraneSource(scale=8)
    assert src.map_for(1400, 0, 0, 10, 10) is None
    assert src.map_for(1400, 0, 0, 10, 10) is None
    assert calls["n"] == 1


def test_membrane_source_out_of_bounds_returns_none(monkeypatch):
    frame = np.full((30, 30, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(mm.pipeline, "load_frame_sam",
                        lambda z, *, scale, frame_store=None: (frame, (0, 0)))
    src = mm.MembraneSource(scale=8)
    assert src.map_for(1400, x0=25, y0=25, h=20, w=20) is None


class _StubSource:
    """Membrane map with a single vertical ridge at column `ridge_x` (full frame)."""
    def __init__(self, ridge_x=15, shape=(50, 50)):
        self.ridge_x, self.shape = ridge_x, shape
    def map_for(self, z, x0, y0, h, w):
        m = np.zeros((h, w), dtype=np.float32)
        col = self.ridge_x - x0
        if 0 <= col < w:
            m[:, col] = 1.0
        return m


def test_score_chain_adds_membrane_columns(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1  # spans ridge at x=15
    d = _write_chain(tmp_path, "AVAL_chain00", {1400: a})
    nbz = {1400: [(15.0, 15.0, "AVAL", "own0")]}  # own node, no foreign
    rec = mm.score_chain(d, "AVAL", nbz, radius=0, membrane_source=_StubSource(15))[0]
    assert rec["spanning_merge"] is True
    assert rec["bled_fraction"] > 0.0
    assert 0.0 <= rec["boundary_on_membrane"] <= 1.0


def test_score_run_reports_mild_bleed_rate(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1  # spans ridge, no foreign node
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({"node_id": ["own0"], "cell_name": ["AVAL"],
                       "z": [1400], "x_tif": [120.0], "y_tif": [120.0]})
    per, summ = mm.score_run(root, annotate_df=df, radius=0, membrane_source=_StubSource(15))
    assert summ["total_foreign_nodes"] == 0
    assert summ["mild_bleed_rate"] == 1.0          # spanning merge with no foreign node
    assert summ["spanning_merge_rate"] == 1.0
    assert "spanning_merge" in per.columns


def test_score_run_no_membrane_keeps_phase0(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({"node_id": ["own0"], "cell_name": ["AVAL"],
                       "z": [1400], "x_tif": [120.0], "y_tif": [120.0]})
    per, summ = mm.score_run(root, annotate_df=df, radius=0, membrane_source=None)
    assert summ["mild_bleed_rate"] is None
    assert per["spanning_merge"].isna().all()
