"""A GUI recrop must persist the new crop window, not just the new masks.

run_chain does not save state; its callers do (batch.py does it right after).
gui._recrop_to_window used to call run_chain and never save, so the masks landed
in the NEW window's space while state.json still described the OLD one. Found on
real data: a recropped chain came back with 1275x1216 masks against a state.json
still claiming 1776x1600, and the GUI then drew the new masks against the old
geometry.

It matters beyond display. The re-propagation driver rebuilds its CropWindow from
state.json, so a stale window sends the whole re-propagation into the wrong space,
and recrop-then-repropagate is the intended workflow.
"""
import json

import pytest

import gui
import pipeline
from sam2_utils import alignment


class _FakeQueue:
    def set_status(self, *a, **k):
        pass


def _chain_on_disk(tmp_path, neuron="AIYR", idx=6):
    d = tmp_path / neuron / f"chain_{idx:02d}"
    (d / "masks").mkdir(parents=True)
    old = alignment.CropWindow(origin_tif=(2976.0, 6344.0), size_tif=(1600, 1776),
                               crop_scale=1, sam_scale=8)
    st = pipeline.ChainState(neuron=neuron, chain_idx=idx)
    st.crop_window = old.to_dict()
    st.anchor_catmaid_z = 1536
    pipeline.save_state(st, d / "state.json")
    return d, old


def _gui_for_recrop(tmp_path, monkeypatch, new_window):
    """A ReviewGUI stand-in wired for _recrop_to_window, with run_chain stubbed."""
    g = gui.ReviewGUI.__new__(gui.ReviewGUI)
    g.neuron, g.chain_idx = "AIYR", 6
    g.data = object()
    g._recrop_picking = False
    g.queue = _FakeQueue()
    g.reviewer = ""
    g.chain = {"cell_name": "AIYR", "nodes": []}
    _, old = _chain_on_disk(tmp_path)
    g._cw = old

    class _Ctx:
        output_root = tmp_path
        cfg = pipeline.PipelineConfig(output_root=tmp_path)
        annotate_df = None
        image_predictor = video_predictor = None

        def ensure_predictors(self, **k):
            pass
    g.ctx = _Ctx()

    def fake_run_chain(state, **kw):
        """Stand in for the real run: set the window as run_chain does, write nothing else."""
        state.crop_window = kw["override_crop_window"].to_dict()
        state.status = "done"
        return state

    monkeypatch.setattr(pipeline, "run_chain", fake_run_chain)
    monkeypatch.setattr(gui.pipeline, "run_chain", fake_run_chain, raising=False)
    monkeypatch.setattr(g, "_close_session", lambda: None)
    monkeypatch.setattr(g, "open_chain", lambda *a, **k: None)
    return g


def test_recrop_writes_the_new_window_to_state_json(tmp_path, monkeypatch):
    new = alignment.CropWindow(origin_tif=(2464.0, 5832.0), size_tif=(2624, 2800),
                               crop_scale=2, sam_scale=8)
    g = _gui_for_recrop(tmp_path, monkeypatch, new)
    g._recrop_to_window(new, "test")

    on_disk = json.loads((tmp_path / "AIYR" / "chain_06" / "state.json").read_text(encoding="utf-8"))
    assert on_disk["crop_window"]["size_tif"] == [2624, 2800], on_disk["crop_window"]
    assert on_disk["crop_window"]["crop_scale"] == 2


def test_the_persisted_window_matches_what_the_masks_were_written_in(tmp_path, monkeypatch):
    """The invariant that was violated: state.json must describe the mask space."""
    new = alignment.CropWindow(origin_tif=(2464.0, 5832.0), size_tif=(2624, 2800),
                               crop_scale=2, sam_scale=8)
    g = _gui_for_recrop(tmp_path, monkeypatch, new)
    g._recrop_to_window(new, "test")

    on_disk = json.loads((tmp_path / "AIYR" / "chain_06" / "state.json").read_text(encoding="utf-8"))
    cw = on_disk["crop_window"]
    w, h = cw["size_tif"]
    implied = (h // cw["crop_scale"], w // cw["crop_scale"])
    assert implied == (1400, 1312), f"state implies {implied}, masks are written 1400x1312"
