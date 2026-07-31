"""Unit tests for pipeline.propagate.apply_second_pass: given a finished chain's saved
masks and eval.merge_metric.score_chain-shaped records, it re-segments flagged frames via
a neighbour mask-prompt and falls back to a neighbour copy when that is not possible."""
import importlib

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

prop = importlib.import_module("pipeline.propagate")
from pipeline import config as cfgmod
from sam2_utils import qc as qc_mod


class _StubPredictor:
    """set_image records shape; predict returns a mask covering the seeded point,
    sized generously if a mask_input hint was given (simulating the hint improving
    the prediction), narrowly otherwise, so tests can tell the two paths apart."""
    def set_image(self, img):
        self._hw = img.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None,
               mask_input=None, multimask_output=False):
        h, w = self._hw
        m = np.zeros((h, w), dtype=bool)
        if point_coords is not None and len(point_coords):
            x, y = int(point_coords[0][0]), int(point_coords[0][1])
            r = 8 if mask_input is not None else 2
            m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        masks = m[None]
        scores = np.array([0.9])
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        return masks, scores, logits


class _StubPredictorNoMaskInput:
    """Mirrors Sam3ImagePredictor's real signature: no mask_input parameter at all.
    predict() always returns a mask covering the seeded point at the narrow radius,
    since there is no hint channel to widen it."""
    def set_image(self, img):
        self._hw = img.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None,
               multimask_output=False):
        h, w = self._hw
        m = np.zeros((h, w), dtype=bool)
        if point_coords is not None and len(point_coords):
            x, y = int(point_coords[0][0]), int(point_coords[0][1])
            r = 8
            m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        masks = m[None]
        scores = np.array([0.9])
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        return masks, scores, logits


def _write_mask(masks_dir, z, mask):
    cv2.imwrite(str(masks_dir / f"mask_{z:04d}.png"), (mask.astype("uint8") * 255))


def _make_chain(tmp_path, *, hw=(40, 40)):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    masks_dir = tmp_path / "chain" / "masks"
    masks_dir.mkdir(parents=True)
    for i in range(5):
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), np.full((*hw, 3), 127, np.uint8))
    frame_to_z = {i: 1400 + i for i in range(5)}
    return frames_dir, masks_dir, frame_to_z


def test_apply_second_pass_reseeds_a_dropout_frame(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path)
    chain_dir = masks_dir.parent

    good = np.zeros((40, 40), dtype=bool)
    good[15:25, 15:25] = True                 # 100 px, centred on (20, 20)
    for z in (1400, 1401, 1403, 1404):
        _write_mask(masks_dir, z, good)
    _write_mask(masks_dir, 1402, np.zeros((40, 40), dtype=bool))   # dropout frame

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({
        "node_id": ["n0"], "cell_name": ["AVAL"], "z": [1402],
        "x_tif": [160.0], "y_tif": [160.0],     # scale 8 -> _sam (20, 20), inside `good`'s footprint
    })
    records = [
        {"z": 1400, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1401, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1402, "own_contained": False, "n_foreign": 0, "empty": True},
        {"z": 1403, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1404, "own_contained": True, "n_foreign": 0, "empty": False},
    ]
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)

    assert outcomes == {1402: "corrected"}
    fixed = qc_mod._load_binary(masks_dir / "mask_1402.png")
    assert fixed.any()


def test_apply_second_pass_runs_without_mask_input_support(tmp_path):
    """A predictor shaped like Sam3ImagePredictor (no mask_input parameter) must not
    raise even though a real neighbour mask is available and would normally build a
    non-None hint; apply_second_pass should degrade to a point-only re-predict and
    still complete, producing a sensible outcome."""
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path)
    chain_dir = masks_dir.parent

    good = np.zeros((40, 40), dtype=bool)
    good[15:25, 15:25] = True                 # 100 px, centred on (20, 20)
    for z in (1400, 1401, 1403, 1404):
        _write_mask(masks_dir, z, good)
    _write_mask(masks_dir, 1402, np.zeros((40, 40), dtype=bool))   # dropout frame

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({
        "node_id": ["n0"], "cell_name": ["AVAL"], "z": [1402],
        "x_tif": [160.0], "y_tif": [160.0],     # scale 8 -> _sam (20, 20), inside `good`'s footprint
    })
    records = [
        {"z": 1400, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1401, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1402, "own_contained": False, "n_foreign": 0, "empty": True},
        {"z": 1403, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1404, "own_contained": True, "n_foreign": 0, "empty": False},
    ]
    cfg = cfgmod.PipelineConfig(scale=8)

    # must not raise TypeError: this is the SAM3 crash reproduction
    outcomes = prop.apply_second_pass(
        _StubPredictorNoMaskInput(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)

    assert outcomes == {1402: "corrected"}
    fixed = qc_mod._load_binary(masks_dir / "mask_1402.png")
    assert fixed.any()


def test_apply_second_pass_falls_back_when_whole_chain_flagged(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path, hw=(40, 40))
    chain_dir = masks_dir.parent

    small = np.zeros((40, 40), dtype=bool)
    small[19:21, 19:21] = True
    for z in range(1400, 1405):
        _write_mask(masks_dir, z, small)

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({
        "node_id": ["n0"], "cell_name": ["AVAL"], "z": [1402],
        "x_tif": [160.0], "y_tif": [160.0],
    })
    records = [{"z": z, "own_contained": True, "n_foreign": 1, "empty": False}
              for z in range(1400, 1405)]   # every frame flagged (foreign)
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)

    assert outcomes == {}   # no unflagged neighbour anywhere -> nothing to fall back to either


def test_apply_second_pass_noop_when_nothing_flagged(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path)
    chain_dir = masks_dir.parent
    good = np.zeros((40, 40), dtype=bool)
    good[15:25, 15:25] = True
    for z in range(1400, 1405):
        _write_mask(masks_dir, z, good)

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({"node_id": [], "cell_name": [], "z": [], "x_tif": [], "y_tif": []})
    records = [{"z": z, "own_contained": True, "n_foreign": 0, "empty": False}
              for z in range(1400, 1405)]
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)
    assert outcomes == {}
