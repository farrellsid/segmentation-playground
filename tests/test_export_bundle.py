"""Export produces a bundle that validates and carries relative frame paths."""
import json

import pytest

import export_bundle
from sam2_utils import bundle


def _tree(tmp_path):
    root = tmp_path / "master"
    for neuron, idx in (("AIAL", 0), ("AIYL", 1)):
        d = root / neuron / f"chain_{idx:02d}"
        (d / "masks").mkdir(parents=True)
        (d / "masks" / "mask_1402.png").write_bytes(b"px")
        (d / "frames").mkdir()
        (d / "frames" / "00000.jpg").write_bytes(b"jpg")
        (d / "state.json").write_text(json.dumps(
            {"neuron": neuron, "chain_idx": idx, "frames_dir": str(d / "frames"),
             "crop_window": {"origin_tif": [0.0, 0.0], "size_tif": [512, 512],
                             "crop_scale": 2, "sam_scale": 8},
             "save_downscale": 8}), encoding="utf-8")
        (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch):
    monkeypatch.setattr(export_bundle.registry, "load_registry",
                        lambda *a, **k: {"AIAL": 42, "AIYL": 7})


def test_export_writes_a_valid_bundle(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    assert bundle.validate_bundle(dest) == []


def test_exported_state_has_a_relative_frames_dir(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    state = json.loads((dest / "AIAL" / "chain_00" / "state.json").read_text(encoding="utf-8"))
    assert state["frames_dir"] == "frames"


def test_export_copies_masks_and_frames(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    assert (dest / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"px"
    assert (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes() == b"jpg"


def test_neuron_filter_limits_the_bundle(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, neurons=["AIAL"], source_tree="master")
    assert (dest / "AIAL").is_dir()
    assert not (dest / "AIYL").exists()


def test_export_writes_meta_with_registry_ids(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    meta = json.loads((dest / "AIYL" / "chain_01" / "meta.json").read_text(encoding="utf-8"))
    assert meta["neuron_id"] == 7
    assert meta["mask_scale"] == 2, "tier-2 chain must record crop_scale, not save_downscale"


def test_export_writes_neurons_csv(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    text = (dest / "neurons.csv").read_text(encoding="utf-8")
    assert "AIAL" in text and "42" in text


def test_export_refuses_a_non_empty_dest_without_force(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    dest.mkdir()
    (dest / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        export_bundle.export_bundle(root, dest, source_tree="master")


def _dead_scratch_tree(tmp_path):
    """A tree whose frames_dir is a dead Narval scratch path, which is the real case."""
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"px")
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0,
         "frames_dir": "/localscratch/4812345/frames/AIAL_chain00_s8",
         "anchor_catmaid_z": 1402, "crop_window": None, "save_downscale": 8}),
        encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    return root


def test_dead_scratch_path_triggers_regeneration(tmp_path, monkeypatch):
    """The real case: every cluster-produced chain records a path that no longer exists."""
    root = _dead_scratch_tree(tmp_path)
    made = tmp_path / "regen"
    made.mkdir()
    (made / "00000.jpg").write_bytes(b"regen")
    calls = []

    def fake_regen(state, **kw):
        calls.append(kw["neuron"])
        return made

    monkeypatch.setattr(export_bundle, "regenerate_frames", fake_regen)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    assert calls == ["AIAL"]
    assert (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes() == b"regen"


def test_existing_bundle_frames_are_not_regenerated(tmp_path, monkeypatch):
    """Export must be resumable: F: drops out and regeneration is the slow part."""
    root = _dead_scratch_tree(tmp_path)
    dest = tmp_path / "bundle"
    (dest / "AIAL" / "chain_00" / "frames").mkdir(parents=True)
    (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").write_bytes(b"already")
    called = []
    monkeypatch.setattr(export_bundle, "regenerate_frames",
                        lambda *a, **k: called.append(1) or tmp_path)
    export_bundle.export_bundle(root, dest, source_tree="master", force=True)
    assert called == []
    assert (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes() == b"already"
