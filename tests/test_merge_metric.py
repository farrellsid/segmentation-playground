import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from eval import merge_metric as mm

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
