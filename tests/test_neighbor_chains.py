"""Unit tests for pipeline.neighbor_chains (the CATMAID neighbor finder).

Torch-free and data-free like test_anchor_select: pipeline imports torch only
lazily, so the pure neighbor-selection logic needs no GPU and no EM stack.

Run either way:
    py -3 -m pytest tests/test_neighbor_chains.py
    py -3 tests/test_neighbor_chains.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import alignment


def _df(rows):
    """rows: list of (node_id, cell_name, z, x_tif, y_tif). x/y duplicated into catmaid x/y."""
    return pd.DataFrame(
        [{"node_id": n, "cell_name": c, "z": z, "x_tif": x, "y_tif": y, "x": x, "y": y}
         for (n, c, z, x, y) in rows]
    )


def test_picks_nearest_other_chains_and_excludes_target():
    # target chain A: nodes 1,2 at z0/z1 near x=100. Neighbors B (x=110, close),
    # C (x=130, farther), D (x=900, far). Same-z so all can contend.
    df = _df([
        (1, "A", 0, 100, 100), (2, "A", 1, 100, 100),
        (3, "B", 0, 110, 100),
        (4, "C", 0, 130, 100),
        (5, "D", 0, 900, 100),
    ])
    chains = [
        {"cell_name": "A", "nodes": [1, 2]},   # idx 0 (target)
        {"cell_name": "B", "nodes": [3]},       # idx 1
        {"cell_name": "C", "nodes": [4]},       # idx 2
        {"cell_name": "D", "nodes": [5]},       # idx 3
    ]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=2,
                                   frame_hw_sam=(1000, 1000))
    assert [o["cell_name"] for o in out] == ["B", "C"]   # nearest two, target excluded
    assert out[0]["chain_idx"] == 1
    assert out[0]["min_dist_sam"] < out[1]["min_dist_sam"]
    assert out[0]["anchor_node_id"] == 3


def test_drops_chains_with_no_in_window_node():
    # target window is the _sam frame 0..50. Neighbor B sits at x=110, outside it.
    df = _df([
        (1, "A", 0, 10, 10),
        (3, "B", 0, 110, 10),
    ])
    chains = [{"cell_name": "A", "nodes": [1]}, {"cell_name": "B", "nodes": [3]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3,
                                   frame_hw_sam=(50, 50))
    assert out == []                              # B is outside the 50x50 frame


def test_requires_shared_z_slice():
    # B only exists on z=5, target only on z=0: cannot contend, so dropped.
    df = _df([(1, "A", 0, 100, 100), (3, "B", 5, 101, 100)])
    chains = [{"cell_name": "A", "nodes": [1]}, {"cell_name": "B", "nodes": [3]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3,
                                   frame_hw_sam=(1000, 1000))
    assert out == []


def test_crop_window_maps_into_pcrop_and_filters():
    # tier-2: a 200x200 _tif window at origin (50,50), crop_scale 1, sam_scale 1.
    # target node at tif (100,100) -> in window. Neighbor at tif (120,120) -> in window;
    # neighbor at tif (300,300) -> outside window, dropped.
    cw = alignment.CropWindow(origin_tif=(50.0, 50.0), size_tif=(200, 200),
                              crop_scale=1, sam_scale=1)
    df = _df([
        (1, "A", 0, 100, 100),
        (3, "B", 0, 120, 120),
        (4, "C", 0, 300, 300),
    ])
    chains = [{"cell_name": "A", "nodes": [1]},
              {"cell_name": "B", "nodes": [3]},
              {"cell_name": "C", "nodes": [4]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3, crop_window=cw)
    assert [o["cell_name"] for o in out] == ["B"]    # C is outside the crop window


# ---------------------------------------------------------------------------
# chain_containing_node: resolve a chain by an anchor node id (naming-agnostic)
# ---------------------------------------------------------------------------

def test_chain_containing_node_returns_the_owning_chain_by_identity():
    chains = [{"cell_name": "A", "nodes": [1, 2]},
              {"cell_name": "B", "nodes": [3, 4]}]
    got = pipeline.chain_containing_node(chains, 4)
    assert got is chains[1]                       # the exact object, so identity checks work


def test_chain_containing_node_matches_int_and_str_node_forms():
    # CATMAID node ids appear as ints and as virtual-node strings; match either form.
    chains = [{"cell_name": "A", "nodes": [25535448, "v_25535450_1", 25535456]}]
    assert pipeline.chain_containing_node(chains, 25535456) is chains[0]   # int query
    assert pipeline.chain_containing_node(chains, "25535448") is chains[0]  # str query
    assert pipeline.chain_containing_node(chains, "v_25535450_1") is chains[0]


def test_chain_containing_node_returns_none_when_absent():
    chains = [{"cell_name": "A", "nodes": [1, 2]}]
    assert pipeline.chain_containing_node(chains, 99) is None
    assert pipeline.chain_containing_node(chains, None) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
