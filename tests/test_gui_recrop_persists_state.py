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
        """Stand in for the real run: set the window as run_chain does, write nothing else.

        frame_to_z is set here because a real run_chain sets it during frame prep,
        after the anchor phase succeeds. Leaving it None would make _recrop_to_window's
        empty-anchor guard (I6) refuse to persist, which is right for a genuinely
        empty anchor but wrong for this stand-in: these tests are about what gets
        WRITTEN on a successful recrop, not about the empty-anchor path (covered in
        test_gui_recrop_in_bundle.py).
        """
        state.crop_window = kw["override_crop_window"].to_dict()
        state.frame_to_z = {0: 1500}
        state.status = "done"
        return state

    # The raw EM guard consults the real config.WORM_PATH, so without this stub these
    # tests pass or fail depending on whether an external drive happens to be mounted.
    # Found for real: the drive dropped off mid-session and all four failed, having
    # passed every run before it. run_chain is stubbed here, so no tif is ever read;
    # what is under test is the bookkeeping around the re-run, not the EM.
    monkeypatch.setattr(pipeline, "raw_em_problem", lambda *a, **k: None)
    monkeypatch.setattr(gui.pipeline, "raw_em_problem", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(pipeline, "run_chain", fake_run_chain)
    monkeypatch.setattr(gui.pipeline, "run_chain", fake_run_chain, raising=False)
    # A real ReviewGUI carries _state (the chain's loaded state.json), and
    # _recrop_to_window reads its config so the re-run reproduces how the chain was
    # segmented rather than the GUI's defaults. None here means "no recorded config",
    # which is the pre-existing behaviour these tests were written against.
    if not hasattr(g, "_state"):
        g._state = None
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


def _cfg_seen_by_run_chain(tmp_path, monkeypatch):
    """Run a recrop and return the PipelineConfig run_chain actually received."""
    seen = {}
    new = alignment.CropWindow(origin_tif=(2464.0, 5832.0), size_tif=(2624, 2800),
                               crop_scale=2, sam_scale=8)
    g = _gui_for_recrop(tmp_path, monkeypatch, new)

    def capture(state, **kw):
        seen["cfg"] = state.config
        state.crop_window = kw["override_crop_window"].to_dict()
        state.frame_to_z = {0: 1500}
        state.status = "done"
        return state

    monkeypatch.setattr(pipeline, "run_chain", capture)
    monkeypatch.setattr(gui.pipeline, "run_chain", capture, raising=False)
    g._recrop_to_window(new, "test")
    return seen["cfg"]


def test_recrop_disables_the_tier2_fallback(tmp_path, monkeypatch):
    """A human-directed recrop must honour the window the human chose.

    chain_crop_fallback is a safety valve for AUTOMATED batch runs: if the crop anchor
    scores poorly, the plain _sam path is probably better than a bad crop. In the GUI the
    premise is inverted, because a person has just drawn or grown this window on purpose.
    Leaving the valve armed made run_chain silently rewrite the chain as _sam, dropping
    crop_window entirely, so the GUI reopened the full low-res frame and the drawn window
    was discarded with no visible error.

    Measured on real data (manual_verify_RIP, 2026-08-25): all three chains recropped in
    one session fell back with reason "score<0.7" and came back with crop_window=None. In
    two of the three the _sam recovery scored LOWER than the crop it replaced (0.64 -> 0.53
    and 0.29 -> 0.23), so the valve also hurt by its own measure. A second-order harm: the
    chain is no longer tier-2, so grow-recrop then refuses it outright.
    """
    cfg = _cfg_seen_by_run_chain(tmp_path, monkeypatch)
    assert cfg.chain_crop is True, "recrop must run the tier-2 crop path"
    assert cfg.chain_crop_fallback is False, (
        "recrop left the batch fallback armed; a poor anchor score will silently discard "
        "the window the reviewer chose and reopen the full _sam frame")


def test_recrop_still_runs_the_crop_path_from_the_given_window(tmp_path, monkeypatch):
    """Guard the rest of the cfg the fix touches, so disabling the valve cannot quietly
    change which window is used or re-enable mask-derived sizing."""
    cfg = _cfg_seen_by_run_chain(tmp_path, monkeypatch)
    assert cfg.chain_crop_from_mask is False, (
        "recrop must size from the reviewer's window, not re-derive one from the mask")
