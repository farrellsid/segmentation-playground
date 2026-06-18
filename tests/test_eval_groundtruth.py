"""Unit tests for eval.groundtruth — VAST metadata parse + labelmap read.

The name-parsing and metadata-table logic run on an inline fixture (portable, no
data drive needed). A real-data smoke test loads one actual GT slice but is skipped
when the GT drive isn't mounted, so the suite is green on any box.

    py -3 -m pytest tests/test_eval_groundtruth.py
    py -3 tests/test_eval_groundtruth.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from eval import groundtruth as G

# A trimmed VAST extended-segmentation file: two comment lines, then rows that
# cover the cases that matter — background, a plain neuron, a bracketed neuron, a
# bracketed neuron with an _or_ name, and a non-swc organelle.
_SAMPLE = """% VAST Lite extended segmentation color file
% Columns: Nr flags ... "name"
0   65537   0 0 0 0   0 0 0 0   1 1 1   0 0 0 0   0   2 2 0 80 80 50   "Background"
1   1   31 145 108 13   160 91 28 15   70 70 19   0 0 0 2   1   28 28 4 400 400 60   "RMDVR_3000154.swc"
2   1   21 16 4 12   2 14 2 7   47 36 13   0 0 1 3   2   40 36 13 480 390 42   "[IL2L]_2998706.swc"
3   1   24 11 4 9   7 4 12 6   58 36 8   0 0 2 4   3   36 26 13 488 480 60   "[PVNL_or_R_1]_2998457.swc"
4   1   24 11 4 9   7 4 12 6   58 36 8   0 0 3 0   4   36 26 13 488 480 60   "phagosome62"
"""


def _md_path():
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / "meta.txt"
    p.write_text(_SAMPLE, encoding="utf-8")
    return p


def test_parse_name_variants():
    assert G.parse_name("RMDVR_3000154.swc") == ("RMDVR", False, 3000154)
    assert G.parse_name("[IL2L]_2998706.swc") == ("IL2L", True, 2998706)
    assert G.parse_name("[PVNL_or_R_1]_2998457.swc") == ("PVNL_or_R_1", True, 2998457)
    assert G.parse_name("phagosome62") == ("phagosome62", False, None)
    assert G.parse_name("object by active zone_2996765.swc") == (
        "object by active zone", False, 2996765)


def test_parse_metadata_table():
    md = G.parse_metadata(_md_path())
    assert list(md.index) == [0, 1, 2, 3, 4]
    assert md.loc[1, "label"] == "RMDVR" and md.loc[1, "kind"] == "neuron"
    assert md.loc[2, "bracketed"] and md.loc[2, "skeleton_id"] == 2998706
    assert md.loc[0, "kind"] == "background"
    assert md.loc[4, "kind"] == "organelle"
    # structural integer columns survive
    assert md.loc[1, "bbox_x2"] == 400 and md.loc[1, "parent"] == 0


def _gt_with_one_slice(label_arr: np.ndarray):
    """Build a GroundTruth over a temp dir holding one labelmap PNG (slice 0)."""
    d = pathlib.Path(tempfile.mkdtemp())
    md = _md_path()
    Image.fromarray(label_arr.astype(np.uint16)).save(
        d / "x.vsseg_export_s000.png")
    return G.GroundTruth.load(md, d, downscale=4)


def test_segment_and_neuron_masks():
    lab = np.array([[0, 1, 1], [2, 3, 4]], dtype=np.uint16)
    gt = _gt_with_one_slice(lab)
    assert gt.n_slices == 1 and gt.slice_indices == [0]
    assert gt.segment_mask(0, 1).sum() == 2
    # neuron lookup by parsed label, not raw name
    assert gt.nr_for_label("RMDVR") == [1]
    assert gt.neuron_mask(0, "RMDVR").sum() == 2
    assert gt.present_segments(0) == {1: 2, 2: 1, 3: 1, 4: 1}
    with pytest.raises(KeyError):
        gt.neuron_mask(0, "NOPE")


def test_neuron_mask_unions_fragments():
    # two segments share a neuron label -> neuron_mask is their union
    sample = _SAMPLE + (
        '5   1   1 1 1 1   1 1 1 1   1 1 1   0 0 4 0   5   1 1 1 9 9 9   "RMDVR_9999.swc"\n')
    d = pathlib.Path(tempfile.mkdtemp())
    mp = d / "meta.txt"; mp.write_text(sample, encoding="utf-8")
    lab = np.array([[1, 5, 0]], dtype=np.uint16)
    Image.fromarray(lab).save(d / "x.vsseg_export_s000.png")
    gt = G.GroundTruth.load(mp, d, downscale=4)
    assert sorted(gt.nr_for_label("RMDVR")) == [1, 5]
    assert gt.neuron_mask(0, "RMDVR").sum() == 2


def test_bbox_in_mask_px_downscaled():
    gt = _gt_with_one_slice(np.zeros((2, 3), np.uint16))
    # metadata bbox for nr 1 is (28,28)-(400,400) full-res; /4 -> (7,7)-(100,100)
    assert gt.bbox_in_mask_px(1) == (7, 7, 100, 100)
    assert gt.bbox_z_range(1) == (4, 60)


# --- real-data smoke test (skips if the GT drive isn't mounted) --------------

def test_real_gt_loads_if_present():
    try:
        from sam2_utils import config
    except Exception:
        pytest.skip("sam2_utils.config not importable")
    if not pathlib.Path(config.GT_METADATA).exists():
        pytest.skip("GT drive not mounted")
    gt = G.GroundTruth.from_config()
    assert gt.n_slices > 0
    assert len(gt.metadata) > 100
    idx = gt.slice_indices[0]
    lab = gt.label_slice(idx)
    assert lab.ndim == 2 and lab.dtype == np.uint16
    # a known confirmed neuron from the real metadata
    assert gt.nr_for_label("RMDVR")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
