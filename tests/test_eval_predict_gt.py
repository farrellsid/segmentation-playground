"""Tests for the plumbing surface of eval.predict_gt: prompt mapping + labelmap
compositing.

    py -3 -m pytest tests/test_eval_predict_gt.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from eval import predict_gt as PG
from eval.registration import Registration


def test_chain_prompt_points_maps_through_registration():
    # identity registration (A = I, no offset): GT px == stack px
    reg = Registration(A=np.eye(2), offsets=np.zeros((1, 2)), z_min=0)
    inp = PG.PredictInputs(
        chains=[], registration=reg, gt=None,
        node_xyz={"1": (10.0, 20.0, 3.0), "v_1_1": (12.0, 22.0, 3.0),
                  "2": (5.0, 5.0, 4.0)},
        grid_hw=(100, 100),
    )
    chain = {"cell_name": "RMDVR!", "nodes": [1, "v_1_1", 2]}
    pts = PG.chain_prompt_points(chain, inp)
    assert set(pts) == {3, 4}                         # grouped by slice z
    assert (10.0, 20.0) in pts[3] and (12.0, 22.0) in pts[3]
    assert pts[4] == [(5.0, 5.0)]


def test_chain_prompt_points_scales_with_registration():
    # A = 1/4 I (the real downscale): stack px 40 -> GT px 10
    reg = Registration(A=np.eye(2) / 4, offsets=np.zeros((1, 2)), z_min=0)
    inp = PG.PredictInputs(
        chains=[], registration=reg, gt=None,
        node_xyz={"9": (40.0, 80.0, 2.0)}, grid_hw=(100, 100))
    pts = PG.chain_prompt_points({"cell_name": "X", "nodes": [9]}, inp)
    assert pts[2] == [(10.0, 20.0)]


def test_composite_labelmaps_ids_and_collisions(tmp_path):
    cfg = PG.PredictGTConfig(pred_dir=tmp_path)
    masks = cfg.masks_dir
    # neuron A: cols 0-2 ; neuron B: cols 2-4 (col 2 overlaps -> 1 collision/row)
    for neu, cols in (("A", (0, 3)), ("B", (2, 5))):
        d = masks / neu; d.mkdir(parents=True)
        m = np.zeros((4, 6), bool); m[:, cols[0]:cols[1]] = True
        Image.fromarray((m.astype(np.uint8) * 255)).save(d / "007.png")

    ids = PG.composite_labelmaps(cfg, (4, 6), [7])
    assert ids == {"A": 1, "B": 2}
    lab = np.asarray(Image.open(cfg.labelmaps_dir / "pred_s007.png"))
    # first-writer-wins: A (id 1) keeps the overlap column 2
    assert (lab[:, 2] == 1).all()
    assert (lab[:, 3] == 2).all() and (lab[:, 0] == 1).all()
    assert (lab[:, 5] == 0).all()                     # background
    saved = json.loads((cfg.labelmaps_dir / "neuron_ids.json").read_text())
    assert saved == {"A": 1, "B": 2}


def test_write_neuron_masks_unions(tmp_path):
    cfg = PG.PredictGTConfig(pred_dir=tmp_path)
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[1, 1] = True
    PG.write_neuron_masks(cfg.pred_dir, "RMDVR", {5: a})
    PG.write_neuron_masks(cfg.pred_dir, "RMDVR", {5: b})   # same slice -> union
    out = np.asarray(Image.open(cfg.masks_dir / "RMDVR" / "005.png")) > 0
    assert out[0, 0] and out[1, 1] and out.sum() == 2


def test_run_erl_dir_labelmaps_source(tmp_path):
    # the pred-mode label source: *_s###.png uint16 read by run_erl
    from eval.run_erl import _DirLabelmaps
    arr = np.zeros((5, 5), dtype=np.uint16); arr[1, 2] = 9
    Image.fromarray(arr, mode="I;16").save(tmp_path / "pred_s012.png")
    src = _DirLabelmaps(tmp_path)
    got = src.slice(12)
    assert got.dtype == np.uint16 and got[1, 2] == 9
    import pytest
    with pytest.raises(KeyError):
        src.slice(999)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
