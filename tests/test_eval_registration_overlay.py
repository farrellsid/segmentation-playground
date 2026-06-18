"""Unit tests for eval.registration_overlay pure data helpers.

Torch/napari-free: only the data plumbing (`_add_overlay_columns`, `nodes_on_slice`,
`nearest_node`) is exercised; the interactive napari `launch` path is not unit-tested
(consistent with gui.py having no UI tests). Run:
    py -3 -m pytest tests/test_eval_registration_overlay.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from eval.registration_overlay import _add_overlay_columns, nearest_node, nodes_on_slice


def _df():
    """A tiny node table: 3 nodes on z=5, 1 on z=6, 1 virtual node on z=5."""
    return pd.DataFrame({
        "node_id": [10, 11, 12, 20, 99],
        "cell_name": ["AVAL", "AVAL", "PVPR", "AVAL", "AVAL"],
        "x": [100.0, 200.0, 300.0, 400.0, 150.0],
        "y": [50.0, 60.0, 70.0, 80.0, 55.0],
        "z": [5.0, 5.0, 5.0, 6.0, 5.0],
        "x_tif": [110.0, 210.0, 310.0, 410.0, 160.0],
        "y_tif": [55.0, 65.0, 75.0, 85.0, 60.0],
        "is_vnode": [False, False, False, False, True],
    })


# --- _add_overlay_columns -----------------------------------------------------

def test_add_overlay_columns_z_int_and_keeps_vnodes():
    out = _add_overlay_columns(_df(), include_vnodes=True)
    assert "z_int" in out.columns
    assert out["z_int"].tolist() == [5, 5, 5, 6, 5]
    assert len(out) == 5                                   # vnode kept


def test_add_overlay_columns_drops_vnodes():
    out = _add_overlay_columns(_df(), include_vnodes=False)
    assert len(out) == 4
    assert 99 not in out["node_id"].tolist()


def test_add_overlay_columns_rounds_fractional_z():
    df = _df()
    df.loc[0, "z"] = 5.4
    df.loc[1, "z"] = 5.6
    out = _add_overlay_columns(df)
    assert out["z_int"].tolist()[:2] == [5, 6]


# --- nodes_on_slice -----------------------------------------------------------

def test_nodes_on_slice_counts_and_ordering():
    df = _add_overlay_columns(_df())
    raw, reg, meta = nodes_on_slice(df, 5)
    assert raw.shape == (4, 3) and reg.shape == (4, 3)     # 3 real + 1 vnode on z=5
    # napari order is (plane, row, col) == (z, y, x)
    assert np.allclose(raw[0], [5.0, 50.0, 100.0])         # z, y, x
    assert np.allclose(reg[0], [5.0, 55.0, 110.0])         # z, y_tif, x_tif
    assert (raw[:, 0] == 5).all()                          # plane coord == requested z
    assert len(meta) == 4 and list(meta.columns) == ["node_id", "cell_name", "x", "y", "z"]


def test_nodes_on_slice_other_plane():
    df = _add_overlay_columns(_df())
    raw, reg, meta = nodes_on_slice(df, 6)
    assert raw.shape == (1, 3)
    assert meta["node_id"].tolist() == [20]


def test_nodes_on_slice_empty():
    df = _add_overlay_columns(_df())
    raw, reg, meta = nodes_on_slice(df, 999)
    assert raw.shape == (0, 3) and reg.shape == (0, 3) and len(meta) == 0


# --- nearest_node -------------------------------------------------------------

def test_nearest_node_in_radius():
    df = _add_overlay_columns(_df())
    _, reg, meta = nodes_on_slice(df, 5)
    # click right next to the registered node 11 at (y_tif, x_tif) = (65, 210)
    hit = nearest_node(meta, reg, (66.0, 211.0), radius_px=30.0)
    assert hit is not None
    assert hit["node_id"] == 11 and hit["cell_name"] == "AVAL"
    assert hit["catmaid_x"] == 200.0 and hit["catmaid_y"] == 60.0 and hit["catmaid_z"] == 5
    assert hit["dist_px"] < 2.0


def test_nearest_node_out_of_radius():
    df = _add_overlay_columns(_df())
    _, reg, meta = nodes_on_slice(df, 5)
    assert nearest_node(meta, reg, (5000.0, 5000.0), radius_px=30.0) is None


def test_nearest_node_empty_slice():
    df = _add_overlay_columns(_df())
    _, reg, meta = nodes_on_slice(df, 999)
    assert nearest_node(meta, reg, (10.0, 10.0), radius_px=30.0) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
