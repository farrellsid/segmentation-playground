"""Compositing a neuron's chains into one volume, then surfacing it.

A neuron's chains sit in different crop windows and overlap in z, so they have to be
placed on one grid before meshing. pipeline.chain_masks_in_sam already returns each
mask on the shared _sam grid with its placement, so this is a paste, not a remap.

The volume is cropped to the neuron's own bounding box. The full _sam grid over a long
neuron is hundreds of MB, which is not a laptop-sized allocation, and the reviewer runs
this on a laptop.

Torch-free, data-free.
    py -3 -m pytest tests/test_render_review_mesh.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


def _tree_with_two_chains(tmp_path):
    """Two chains of one neuron, in different places, overlapping in z."""
    root = tmp_path / "tree"
    root.mkdir(parents=True)
    (root / "_manifest.csv").write_text("neuron,chain_idx\n", encoding="utf-8")
    placements = {}
    for ci, (x0, y0, zs) in enumerate([(100, 200, [1500, 1501, 1502]),
                                       (140, 210, [1502, 1503])]):
        d = root / "AIBL" / f"chain_{ci:02d}"
        (d / "masks").mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"neuron": "AIBL", "chain_idx": ci}),
                                      encoding="utf-8")
        blocks = {}
        for z in zs:
            m = np.zeros((6, 6), dtype=bool)
            m[1:5, 1:5] = True
            blocks[z] = (m, x0, y0)
        placements[ci] = blocks
    return root, placements


def _patch_masks(monkeypatch, placements):
    def fake(chain_dir):
        ci = int(pathlib.Path(chain_dir).name.split("_")[-1])
        return placements.get(ci, {})
    monkeypatch.setattr(render_review.pipeline, "chain_masks_in_sam", fake)


class TestNeuronVolume:
    def test_composites_every_chain_onto_one_grid(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, spacing = render_review.neuron_volume(root, "AIBL")
        assert vol.ndim == 3
        assert vol.sum() > 0
        # 4 z values across both chains
        assert vol.shape[0] == 4, f"expected 4 z planes, got {vol.shape[0]}"

    def test_volume_is_cropped_to_the_neuron_bbox_not_the_full_frame(self, tmp_path,
                                                                    monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, _ = render_review.neuron_volume(root, "AIBL")
        # masks span x 101..144 and y 201..214, so the bbox is tens of px, not 1152
        assert vol.shape[1] < 100 and vol.shape[2] < 100, (
            f"volume is {vol.shape}, which is the full grid rather than the bbox")

    def test_spacing_is_z_finer_than_xy(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        _vol, spacing = render_review.neuron_volume(root, "AIBL")
        assert spacing == (50.0, 128.0, 128.0)
        assert spacing[0] < spacing[1], "z must be the finer axis at scale 8"

    def test_refuses_a_volume_over_the_memory_budget(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        with pytest.raises(SystemExit) as e:
            render_review.neuron_volume(root, "AIBL", max_voxels=10)
        assert "voxel" in str(e.value).lower()


class TestNeuronMesh:
    def test_writes_a_ply_that_parses(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        out = render_review.neuron_mesh(root, "AIBL", tmp_path / "AIBL.ply")
        assert out is not None and out.exists()
        text = out.read_text(encoding="utf-8")
        assert text.startswith("ply")
        n_verts = int([ln for ln in text.splitlines()
                       if ln.startswith("element vertex")][0].split()[-1])
        assert n_verts > 0

    def test_a_neuron_with_no_masks_returns_none_instead_of_raising(self, tmp_path,
                                                                    monkeypatch):
        root, _p = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, {})
        assert render_review.neuron_mesh(root, "AIBL", tmp_path / "x.ply") is None
        assert not (tmp_path / "x.ply").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
