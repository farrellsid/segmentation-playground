"""Torch-free tests for the labelmap composer + crop-aware _sam placement (eval.score_labelmap)."""
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval.score_labelmap import chain_sam_mask, SamLabelComposer


def _write_mask(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(arr).astype(np.uint8) * 255)).save(path)


def test_chain_sam_mask_noncrop_passthrough(tmp_path):
    sam_hw = (40, 50)
    m = np.zeros(sam_hw, bool); m[5:10, 6:12] = True
    p = tmp_path / "mask_0100.png"; _write_mask(p, m)
    out = chain_sam_mask(p, None, sam_hw)
    assert out.shape == sam_hw and np.array_equal(out, m)


def test_chain_sam_mask_pcrop_placement(tmp_path):
    # sam frame 100x100, downscale 8. window: origin_tif=(80,160)->_sam (10,20);
    # size_tif=(160,240)->_sam (20,30); crop_scale 2 -> _pcrop mask (120,80) all-ones.
    sam_hw = (100, 100)
    cw = {"origin_tif": [80, 160], "size_tif": [160, 240], "crop_scale": 2, "sam_scale": 8}
    pcrop = np.ones((120, 80), bool)
    p = tmp_path / "mask_0007.png"; _write_mask(p, pcrop)
    out = chain_sam_mask(p, cw, sam_hw)
    assert out.shape == sam_hw
    # the window occupies _sam rows [20,50) cols [10,30)
    assert out[20:50, 10:30].all()
    out[20:50, 10:30] = False
    assert not out.any()                       # nothing outside the window


def test_composer_first_writer_wins_and_ids(tmp_path):
    sam_hw = (20, 20)
    # neuron A (id 1) and B (id 2) overlap on slice 3; A wins the overlap.
    a = np.zeros(sam_hw, bool); a[2:8, 2:8] = True
    b = np.zeros(sam_hw, bool); b[5:11, 5:11] = True       # overlaps A in [5:8,5:8]
    _write_mask(tmp_path / "A" / "chain_00" / "masks" / "mask_0003.png", a)
    _write_mask(tmp_path / "B" / "chain_00" / "masks" / "mask_0003.png", b)
    comp = SamLabelComposer(tmp_path, ["A", "B"], sam_hw)
    assert comp.neuron_ids == {"A": 1, "B": 2}
    lab, collisions = comp.labelmap(3)
    assert collisions == 9                                  # the 3x3 overlap
    assert (lab[2:8, 2:8] == 1).all()                      # A kept its full block
    assert lab[8:11, 8:11].max() == 2 and (lab == 2).any() # B owns its non-overlap
    assert lab[5:8, 5:8].max() == 1                         # first-writer (A) wins overlap


def test_composer_unions_chains(tmp_path):
    sam_hw = (20, 20)
    a1 = np.zeros(sam_hw, bool); a1[1:4, 1:4] = True
    a2 = np.zeros(sam_hw, bool); a2[10:13, 10:13] = True
    _write_mask(tmp_path / "A" / "chain_00" / "masks" / "mask_0005.png", a1)
    _write_mask(tmp_path / "A" / "chain_01" / "masks" / "mask_0005.png", a2)
    comp = SamLabelComposer(tmp_path, ["A"], sam_hw)
    m = comp.neuron_mask("A", 5)
    assert m[1:4, 1:4].all() and m[10:13, 10:13].all()     # both chains unioned
    assert m.sum() == 18


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
