"""A cleared frames_dir means 'regenerate', not 'crash'.

import_bundle clears the master's frames_dir when a chain comes home recropped, because
the recorded view directory would otherwise be reused with the OLD window's frames. That
makes None a state the GUI now reaches on a normal path.

Two layers of test here:

1. gui._ensure_local_frames, the pure "prepare or reuse" step, called directly. This
   is the ORIGINAL test in this file, and it passed even while the feature was broken:
   it bypasses the caller guard entirely, so it never exercised the actual defect.
2. gui._resolve_chain_frames, the caller-side guard open_chain actually runs. The real
   bug lived in ITS condition (`if state.frames_dir and ...`, which treats a cleared
   frames_dir as "nothing to do" instead of "regenerate"), not in _ensure_local_frames.
   This is the regression test: it fails against the old guard and passes only once
   the guard calls through on a None frames_dir.

    py -3 -m pytest tests/test_ensure_local_frames_none.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


def test_a_cleared_frames_dir_regenerates_instead_of_raising(monkeypatch, tmp_path):
    prepared = {}

    def _prepare(chain, annotate_df, *, scale, frames_root, anchor_catmaid_z,
                 neuron, chain_idx, z_range=None):
        prepared["called"] = True
        return str(tmp_path / "fresh"), {0: 1500}, 0, 1

    monkeypatch.setattr(gui.pipeline, "prepare_video_frames", _prepare)
    frames_dir, frame_to_z, anchor_idx = gui._ensure_local_frames(
        None, {0: 1500}, 0, chain={}, cw=None,
        cfg=types.SimpleNamespace(scale=8, frames_root=tmp_path),
        annotate_df=None, anchor_catmaid_z=1500, neuron="AIAL", chain_idx=0,
        chain_dir=tmp_path)
    assert prepared["called"] is True
    assert frame_to_z == {0: 1500} and anchor_idx == 0


def test_the_caller_guard_regenerates_a_state_with_no_frames_dir(monkeypatch, tmp_path):
    """C2 regression test: exercises _resolve_chain_frames, the caller open_chain
    actually calls, not _ensure_local_frames in isolation. Before the fix, the
    guard's condition was `if state.frames_dir and state.anchor_catmaid_z is not
    None`, so a cleared (None) frames_dir short-circuited the whole block and
    returned (None, None, None) without ever calling _ensure_local_frames. That
    left review.load_chain to fall back to state["frames_dir"], which is None,
    stringify it to the literal text "None", and fail with FileNotFoundError deep
    inside a JPEG read: the exact failure the reviewer hit opening a recropped
    chain that had come home through import_bundle.
    """
    prepared = {}

    def _prepare(chain, annotate_df, *, scale, frames_root, anchor_catmaid_z,
                 neuron, chain_idx, z_range=None):
        prepared["called"] = True
        return str(tmp_path / "fresh"), {0: 1500}, 0, 1

    monkeypatch.setattr(gui.pipeline, "prepare_video_frames", _prepare)
    state = types.SimpleNamespace(frames_dir=None, frame_to_z=None, anchor_frame_idx=None,
                                  anchor_catmaid_z=1500)
    frames_dir, frame_to_z, anchor_idx = gui._resolve_chain_frames(
        state, chain={}, cw=None,
        cfg=types.SimpleNamespace(scale=8, frames_root=tmp_path),
        annotate_df=None, neuron="AIAL", chain_idx=0, anchor_only=False,
        context_frames=0, chain_dir=tmp_path)
    assert prepared["called"] is True, (
        "the guard skipped _ensure_local_frames instead of regenerating from a "
        "cleared frames_dir")
    assert frame_to_z == {0: 1500} and anchor_idx == 0


def test_the_caller_guard_does_nothing_without_a_state(tmp_path):
    """No state.json at all (a chain that was never run) is a different case from a
    cleared frames_dir, and must stay a no-op: there is nothing to regenerate from."""
    assert gui._resolve_chain_frames(
        None, chain={}, cw=None, cfg=types.SimpleNamespace(scale=8, frames_root=tmp_path),
        annotate_df=None, neuron="AIAL", chain_idx=0, anchor_only=False,
        context_frames=0, chain_dir=tmp_path) == (None, None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
