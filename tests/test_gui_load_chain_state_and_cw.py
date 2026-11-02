"""_load_chain_state_and_cw: the (chain, state, crop_window) lookup open_chain
and the background prefetch (Task 3) both need, factored out so the two call
sites can't drift. Mirrors the exact short-circuit order open_chain used to
inline: a chain this session's chain list does not have returns (None, None,
None) WITHOUT ever touching state.json, so a scope mismatch stays a clean
"not in this session" signal even if state.json on disk is malformed.

    py -3 -m pytest tests/test_gui_load_chain_state_and_cw.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import gui


def _ctx(*, find_chain):
    return types.SimpleNamespace(find_chain=find_chain)


def test_chain_not_in_session_returns_all_none_without_reading_state(tmp_path):
    chain_dir = tmp_path / "AIZL" / "chain_03"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text("not valid json {{{")   # would raise if read
    ctx = _ctx(find_chain=lambda neuron, idx: None)

    assert gui._load_chain_state_and_cw(ctx, "AIZL", 3, chain_dir) == (None, None, None)


def test_tier2_chain_returns_chain_state_and_crop_window(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_00"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIAL", "chain_idx": 0, "status": "done",
        "anchor_catmaid_z": 1500, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "frames"), "frame_to_z": {"0": 1500}, "n_frames": 1,
        "crop_window": {"origin_tif": [100.0, 200.0], "size_tif": [400, 300],
                        "crop_scale": 2, "sam_scale": 8},
    }))
    stub_chain = {"nodes": ["a", "b"]}
    ctx = _ctx(find_chain=lambda neuron, idx: stub_chain)

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 0, chain_dir)

    assert chain is stub_chain
    assert state.anchor_catmaid_z == 1500
    assert cw.origin_tif == (100.0, 200.0) and cw.crop_scale == 2


def test_legacy_sam_chain_has_no_crop_window(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_01"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIAL", "chain_idx": 1, "status": "done",
        "anchor_catmaid_z": 1500, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "frames"), "frame_to_z": {"0": 1500}, "n_frames": 1,
        "crop_window": None,
    }))
    ctx = _ctx(find_chain=lambda neuron, idx: {"nodes": []})

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 1, chain_dir)

    assert chain is not None and state is not None
    assert cw is None


def test_chain_in_session_but_no_state_json_yet(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_02"
    chain_dir.mkdir(parents=True)   # chain dir exists, never run: no state.json
    ctx = _ctx(find_chain=lambda neuron, idx: {"nodes": []})

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 2, chain_dir)

    assert chain is not None
    assert state is None and cw is None


def test_open_chain_preserves_state_and_cw_on_failed_lookup(tmp_path):
    """Regression test: when open_chain fails due to a chain not being in this
    session's scope, self._state and self._cw must NOT be cleared to None (they
    should stay at whatever the previous successful open_chain set them to).
    Before the fix, the tuple unpacking happened before the None check, wiping
    them unconditionally."""
    root = tmp_path / "tree"
    chain_dir = root / "AIZL" / "chain_03"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIZL", "chain_idx": 3, "status": "done",
        "anchor_catmaid_z": 1524, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "does_not_exist"),
        "frame_to_z": {"0": 1524}, "n_frames": 1,
        "crop_window": {"origin_tif": [3888.0, 6216.0], "size_tif": [1168, 1328],
                        "crop_scale": 2, "sam_scale": 8},
    }), encoding="utf-8")

    # Pre-seed with sentinel objects (simulating a previous successful open_chain)
    sentinel_state = types.SimpleNamespace(anchor_catmaid_z=9999)
    sentinel_cw = types.SimpleNamespace(size_tif=[99, 99], crop_scale=99)

    fake_self = types.SimpleNamespace(
        ctx=types.SimpleNamespace(
            output_root=root,
            find_chain=lambda neuron, idx: None,   # not in this session
            annotate_df=None,
            cfg=types.SimpleNamespace(frames_root=tmp_path / "frames_cache"),
        ),
        _close_session=lambda: None,
        anchor_only=False,
        context_frames=0,
        _state=sentinel_state,
        _cw=sentinel_cw,
    )

    # open_chain should raise but NOT modify _state or _cw
    with pytest.raises(LookupError, match="AIZL"):
        gui.ReviewGUI.open_chain(fake_self, "AIZL", 3)

    # _state and _cw should still be the sentinel objects, not None
    assert fake_self._state is sentinel_state
    assert fake_self._cw is sentinel_cw


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
