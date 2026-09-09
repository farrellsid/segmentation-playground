"""open_chain must fail clearly when the chain isn't in this session's chain list.

Real crash (Lucinda, opening AIZL chain_03 from a session whose chains.json /
neuron scope did not include AIZL): ctx.find_chain returns None, open_chain
stores it as self.chain unchecked, and the crash surfaces three calls later as
`TypeError: 'NoneType' object is not subscriptable` deep inside
prepare_chain_crop_frames's `for n in chain["nodes"]`. That is confusing at the
point it happens and useless for diagnosing the actual problem (a scope/session
mismatch, not a missing file). open_chain should refuse immediately, at the
point the mismatch is known, with a message naming the chain.
"""
from __future__ import annotations

import json
import types

import pytest

import gui


def test_open_chain_raises_clearly_when_find_chain_returns_none(tmp_path):
    root = tmp_path / "tree"
    chain_dir = root / "AIZL" / "chain_03"
    chain_dir.mkdir(parents=True)
    # A real, complete state.json (tier-2, stale frames_dir) - the exact shape
    # that reached prepare_chain_crop_frames in the real crash.
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIZL", "chain_idx": 3, "status": "done",
        "anchor_catmaid_z": 1524, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "does_not_exist"),
        "frame_to_z": {"0": 1524}, "n_frames": 1,
        "crop_window": {"origin_tif": [3888.0, 6216.0], "size_tif": [1168, 1328],
                        "crop_scale": 2, "sam_scale": 8},
    }), encoding="utf-8")

    fake_self = types.SimpleNamespace(
        ctx=types.SimpleNamespace(
            output_root=root,
            find_chain=lambda neuron, idx: None,   # not in this session's chain list
            annotate_df=None,
            cfg=types.SimpleNamespace(frames_root=tmp_path / "frames_cache"),
        ),
        _close_session=lambda: None,
        anchor_only=False,
        context_frames=0,
    )

    with pytest.raises(LookupError, match="AIZL"):
        gui.ReviewGUI.open_chain(fake_self, "AIZL", 3)
