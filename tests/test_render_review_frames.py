"""The one place a bundle and an output tree genuinely differ.

Both are laid out <neuron>/chain_NN/state.json, so bundle.index_chains and
pipeline.chain_masks_in_sam already read either. What differs is EM: a bundle carries
its own per-chain frames/, an output tree does not and must go to the EM store. The
reviewer's Mac has no EM store at all, so the bundle branch has to work with no
network and no F: drive.

Torch-free and data-free: the tree branch stubs pipeline.load_frame_sam.
    py -3 -m pytest tests/test_render_review_frames.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


def _chain(tmp_path, kind, n=3):
    root = tmp_path / kind
    d = root / "AIBL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    state = {"neuron": "AIBL", "chain_idx": 0,
             "frame_to_z": {str(i): 1500 + i for i in range(n)}}
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if kind == "bundle":
        (d / "frames").mkdir()
        for i in range(n):
            img = np.full((20, 24, 3), 10 * (i + 1), dtype=np.uint8)
            cv2.imwrite(str(d / "frames" / f"{i:05d}.jpg"), img)
        (root / "bundle.json").write_text('{"schema_version": 1, "chains": []}',
                                          encoding="utf-8")
    else:
        (root / "_manifest.csv").write_text("neuron,chain_idx\nAIBL,0\n", encoding="utf-8")
    return root, d, state


class TestSourceKind:
    def test_bundle_json_means_bundle(self, tmp_path):
        root, _d, _s = _chain(tmp_path, "bundle")
        assert render_review.source_kind(root) == "bundle"

    def test_manifest_means_tree(self, tmp_path):
        root, _d, _s = _chain(tmp_path, "tree")
        assert render_review.source_kind(root) == "tree"

    def test_neither_is_refused_naming_what_was_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit) as e:
            render_review.source_kind(tmp_path / "empty")
        assert "bundle.json" in str(e.value) and "_manifest.csv" in str(e.value)


class TestChainFrames:
    def test_bundle_reads_its_own_frames_without_the_em_store(self, tmp_path, monkeypatch):
        root, d, state = _chain(tmp_path, "bundle")

        def explode(*a, **k):
            raise AssertionError("bundle path must not touch the EM store")

        monkeypatch.setattr(render_review.pipeline, "load_frame_sam", explode)
        frames = render_review.chain_frames(d, state, "bundle")
        assert sorted(frames) == [0, 1, 2]
        assert frames[0].shape == (20, 24, 3)
        assert frames[1].mean() > frames[0].mean(), "frames came back in the wrong order"

    def test_tree_reads_the_em_store_at_each_z(self, tmp_path, monkeypatch):
        root, d, state = _chain(tmp_path, "tree")
        seen = []

        def fake_load(z, *, scale):
            seen.append(int(z))
            return np.full((30, 30, 3), int(z) % 255, dtype=np.uint8), (240, 240)

        monkeypatch.setattr(render_review.pipeline, "load_frame_sam", fake_load)
        frames = render_review.chain_frames(d, state, "tree")
        assert seen == [1500, 1501, 1502], "must request the chain's own z values"
        assert sorted(frames) == [0, 1, 2]

    def test_bundle_chain_with_no_frames_dir_names_the_chain(self, tmp_path):
        """A silently skipped chain is how a video stops being of the whole neuron
        while still looking finished."""
        root, d, state = _chain(tmp_path, "bundle")
        import shutil
        shutil.rmtree(d / "frames")
        with pytest.raises(SystemExit) as e:
            render_review.chain_frames(d, state, "bundle")
        assert "chain_00" in str(e.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
