"""A cleared frames_dir means 'regenerate', not 'crash'.

import_bundle clears the master's frames_dir when a chain comes home recropped, because
the recorded view directory would otherwise be reused with the OLD window's frames. That
makes None a state the GUI now reaches on a normal path.

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
