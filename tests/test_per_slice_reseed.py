import importlib

import numpy as np
import pandas as pd
from pipeline import predict, PipelineConfig
from pipeline import config as cfgmod
from sam2_utils.alignment import CropWindow

# pipeline/__init__.py re-exports propagate() (the function) under the name
# "propagate", which shadows the submodule of the same name on the pipeline
# package object. `from pipeline import propagate` would therefore hand back
# the function, not the module segment_per_slice lives in, so fetch the
# submodule straight from the import machinery instead.
prop = importlib.import_module("pipeline.propagate")


def test_centreline_by_z_uses_nodes_and_interpolates_gaps():
    # chain nodes at z=1400 (x_tif=80,y_tif=800) and z=1402 (x_tif=120,y_tif=840); z=1401 is a gap
    chain = {"cell_name": "AVAL", "nodes": ["n0", "n2"]}
    df = pd.DataFrame({
        "node_id": ["n0", "n2", "other"],
        "cell_name": ["AVAL", "AVAL", "AVBR"],
        "z": [1400, 1402, 1401],
        "x_tif": [80.0, 120.0, 9999.0],   # 'other' is a different neuron, must be ignored
        "y_tif": [800.0, 840.0, 9999.0],
    })
    got = predict.centreline_by_z(chain, df)
    assert set(got) == {1400, 1401, 1402}
    assert got[1400] == (80.0, 800.0)
    assert got[1402] == (120.0, 840.0)
    # z=1401 interpolated halfway between the two chain nodes, NOT the foreign node
    assert got[1401] == (100.0, 820.0)


def test_per_slice_reseed_flag_defaults_false():
    assert PipelineConfig().per_slice_reseed is False


class _StubPredictor:
    """Minimal image-predictor stub: set_image records shape, predict returns one
    canned mask covering the seeded point, so segment_per_slice runs torch-free.

    logits are shaped (n_masks, 256, 256), SAM2's actual low_res_masks resolution,
    deliberately DIFFERENT from the full-res mask shape (h, w): segment_per_slice must
    derive frame_conf's foreground from the logits' own threshold, never by indexing
    the low-res logits with the full-res mask (that raises IndexError on real SAM2).
    """
    def set_image(self, img): self._hw = img.shape[:2]
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        h, w = self._hw
        m = np.zeros((h, w), dtype=bool)
        if point_coords is not None and len(point_coords):
            x, y = int(point_coords[0][0]), int(point_coords[0][1])
            m[max(0, y-2):y+3, max(0, x-2):x+3] = True
        masks = np.stack([m, m, m]) if multimask_output else m[None]
        scores = np.array([0.9, 0.8, 0.7][: masks.shape[0]])
        logits = np.zeros((masks.shape[0], 256, 256), dtype=np.float32)
        return masks, scores, logits


class _RecordingStubPredictor(_StubPredictor):
    """Same canned-mask behaviour as _StubPredictor, but records the point_labels
    (and point_coords) segment_per_slice actually hands to predict(), so a test
    can assert what did or did not survive the crop-window filter."""
    def __init__(self):
        self.seen_labels = []
        self.seen_coords = []

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        self.seen_coords.append(None if point_coords is None else np.array(point_coords))
        self.seen_labels.append(None if point_labels is None else np.array(point_labels))
        return super().predict(point_coords=point_coords, point_labels=point_labels,
                               box=box, multimask_output=multimask_output)


def test_segment_per_slice_drops_out_of_window_negatives(tmp_path):
    # cw set (a small crop window) + a negative neighbour node far enough away
    # that its _pcrop coordinate lands outside the crop; the positive anchor's
    # _pcrop coordinate lands inside. segment_per_slice must drop the negative
    # before predict() sees it, mirroring anchor_crop_predict's
    # `in_bounds | (labels == 1)` filter.
    import cv2
    fdir = tmp_path / "frames"; fdir.mkdir()
    cv2.imwrite(str(fdir / "00000.jpg"), np.full((40, 40, 3), 127, np.uint8))
    frame_to_z = {0: 1400}
    centreline_tif = {1400: (160.0, 100.0)}   # -> _sam (20, 12.5) -> _crop (16, 10): in-window

    df = pd.DataFrame({
        "node_id": ["n0", "n1"],
        "cell_name": ["AVAL", "AVAL"],
        "z": [1400, 1400],
        "x": [0.0, 100.0],          # CATMAID xy, used only to rank same-z neighbours
        "y": [0.0, 100.0],
        "x_tif": [160.0, 100000.0],  # n1 maps far outside the crop window
        "y_tif": [100.0, 100000.0],
    })
    cfg = cfgmod.PipelineConfig(scale=8, k_max_neg=1)
    cw = CropWindow(origin_tif=(0.0, 0.0), size_tif=(400, 400), crop_scale=10, sam_scale=8)

    stub = _RecordingStubPredictor()
    vs, conf, piou = prop.segment_per_slice(
        stub, str(fdir), frame_to_z, centreline_tif, df,
        cfg=cfg, obj_id=1, cw=cw)

    assert len(stub.seen_labels) == 1
    labels_seen = stub.seen_labels[0]
    assert labels_seen is not None
    assert list(labels_seen) == [1]           # the out-of-window negative never reached predict()
    assert 1 in vs[0] and vs[0][1].sum() > 0   # the positive still seeded a mask


def test_do_segmentation_routes_on_per_slice_reseed_flag(monkeypatch):
    # _do_segmentation is the module-level dispatch run_chain calls; verify the
    # flag alone decides segment_per_slice vs. propagate, with everything else stubbed.
    import pipeline.orchestrator as orch

    calls = {"per_slice": 0, "propagate": 0}

    def fake_per_slice(*a, **k):
        calls["per_slice"] += 1
        return ({0: {1: np.zeros((4, 4), bool)}}, {0: 0.0}, {0: 0.0})

    def fake_propagate(*a, **k):
        calls["propagate"] += 1
        return ({}, {}, {})

    monkeypatch.setattr(orch, "segment_per_slice", fake_per_slice)
    monkeypatch.setattr(orch, "propagate", fake_propagate)

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    df = pd.DataFrame({"node_id": ["n0"], "cell_name": ["AVAL"], "z": [1400],
                       "x_tif": [10.0], "y_tif": [10.0]})
    common = dict(image_predictor=None, video_predictor=None, frames_dir="unused",
                 frame_to_z={0: 1400}, prompts=None, anchor_frame_idx=0,
                 chain=chain, annotate_df=df, cw=None, obj_id=1)

    orch._do_segmentation(cfgmod.PipelineConfig(per_slice_reseed=True), **common)
    assert calls == {"per_slice": 1, "propagate": 0}

    orch._do_segmentation(cfgmod.PipelineConfig(per_slice_reseed=False), **common)
    assert calls == {"per_slice": 1, "propagate": 1}


def test_segment_per_slice_returns_a_mask_per_frame(tmp_path):
    # 3 frames on disk, full-frame (cw=None -> _sam space)
    import cv2
    fdir = tmp_path / "frames"; fdir.mkdir()
    for i in range(3):
        cv2.imwrite(str(fdir / f"{i:05d}.jpg"), np.full((40, 40, 3), 127, np.uint8))
    frame_to_z = {0: 1400, 1: 1401, 2: 1402}
    centreline_tif = {1400: (80.0, 80.0), 1401: (80.0, 80.0), 1402: (80.0, 80.0)}  # scale 8 -> (10,10)
    df = pd.DataFrame({"node_id": [], "cell_name": [], "z": [], "x_tif": [], "y_tif": []})
    cfg = cfgmod.PipelineConfig(scale=8, k_max_neg=0)
    vs, conf, piou = prop.segment_per_slice(
        _StubPredictor(), str(fdir), frame_to_z, centreline_tif, df,
        cfg=cfg, obj_id=1, cw=None)
    assert set(vs) == {0, 1, 2}
    assert all(1 in vs[f] for f in vs)              # obj_id present per frame
    assert vs[0][1].sum() > 0                        # canned mask non-empty at the seed
    assert set(conf) == {0, 1, 2} and set(piou) == {0, 1, 2}
