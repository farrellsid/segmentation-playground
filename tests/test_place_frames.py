"""place_frames copies a drive's frames into a git clone that has none."""
import json

import pytest

import place_frames
from sam2_utils import bundle


def _clone(tmp_path, chains=(("AIAL", 0), ("AIAL", 1))):
    """A bundle as git carries it: masks and metadata, no frames."""
    root = tmp_path / "clone"
    for neuron, idx in chains:
        d = root / neuron / f"chain_{idx:02d}"
        (d / "masks").mkdir(parents=True)
        (d / "masks" / "mask_1402.png").write_bytes(b"px")
        (d / "state.json").write_text(json.dumps(
            {"neuron": neuron, "chain_idx": idx, "frames_dir": "frames",
             "n_frames": 2, "crop_window": None, "save_downscale": 8}), encoding="utf-8")
        (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({
            "schema_version": 1, "neuron_id": 42, "cell_name": neuron, "chain_idx": idx,
            "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
            "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "chains.json").write_text("[]", encoding="utf-8")
    (root / "data" / "nodes.csv").write_text("node_id\n1\n", encoding="utf-8")
    man = bundle.build_manifest(bundle.index_chains(root), source_tree="t")
    (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    return root


def _source(tmp_path, chains=(("AIAL", 0), ("AIAL", 1)), n=2):
    """The drive copy, which has the frames.

    It carries a manifest because the real one does: place_frames checks for it on
    both sides, so pointing at the wrong folder fails immediately instead of
    scattering frames somewhere nothing reads them from.
    """
    root = tmp_path / "drive"
    for neuron, idx in chains:
        f = root / neuron / f"chain_{idx:02d}" / "frames"
        f.mkdir(parents=True)
        for i in range(n):
            (f / f"{i:05d}.jpg").write_bytes(f"{neuron}{idx}-{i}".encode())
    (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
        {"schema_version": bundle.BUNDLE_SCHEMA_VERSION, "source_tree": "t",
         "chains": []}), encoding="utf-8")
    return root


def test_frames_land_in_every_chain(tmp_path):
    clone, drive = _clone(tmp_path), _source(tmp_path)
    result = place_frames.place_frames(clone, drive)
    assert result["placed"] == 2
    assert (clone / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes() == b"AIAL0-0"
    assert (clone / "AIAL" / "chain_01" / "frames" / "00001.jpg").read_bytes() == b"AIAL1-1"


def test_the_clone_validates_afterwards(tmp_path):
    """The point of the exercise: a clone plus frames is a usable bundle."""
    clone, drive = _clone(tmp_path), _source(tmp_path)
    assert bundle.validate_bundle(clone), "should be missing frames before"
    place_frames.place_frames(clone, drive)
    assert bundle.validate_bundle(clone) == []


def test_already_placed_chains_are_skipped(tmp_path):
    clone, drive = _clone(tmp_path), _source(tmp_path)
    place_frames.place_frames(clone, drive)
    again = place_frames.place_frames(clone, drive)
    assert again["placed"] == 0 and again["already"] == 2


def test_a_chain_the_drive_lacks_is_reported_not_crashed(tmp_path):
    """Half a delivery is the realistic accident, and she needs to be told which half."""
    clone = _clone(tmp_path)
    drive = _source(tmp_path, chains=(("AIAL", 0),))
    result = place_frames.place_frames(clone, drive)
    assert result["placed"] == 1
    assert result["missing"] == ["AIAL/chain_01"]


def test_dry_run_writes_nothing(tmp_path):
    clone, drive = _clone(tmp_path), _source(tmp_path)
    result = place_frames.place_frames(clone, drive, dry_run=True)
    assert result["placed"] == 2
    assert not (clone / "AIAL" / "chain_00" / "frames").exists()


def test_a_source_that_is_not_a_bundle_fails_loudly(tmp_path):
    clone = _clone(tmp_path)
    empty = tmp_path / "not_a_bundle"
    empty.mkdir()
    with pytest.raises(SystemExit):
        place_frames.place_frames(clone, empty)
