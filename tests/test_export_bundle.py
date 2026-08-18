"""Export produces a bundle that validates and carries relative frame paths."""
import json
import sys
import types
from pathlib import Path

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


def _partial_tree(tmp_path):
    """A chain that expects 5 frames but only frame 0 made it to disk: the exact
    shape of an export killed mid-copytree, the F: drop-out resumability exists for.
    """
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"px")
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0,
         "frames_dir": "/localscratch/4812345/frames/AIAL_chain00_s8",
         "anchor_catmaid_z": 1402, "crop_window": None, "save_downscale": 8,
         "n_frames": 5}),
        encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    return root


def test_partial_export_frame_zero_only_is_resumed_not_skipped(tmp_path, monkeypatch):
    """The resume hole: an export interrupted mid-copytree leaves frame 0 on disk
    with the other 4 (of the recorded n_frames=5) missing. Presence of frame 0
    alone used to be the whole resume check, so this chain would be silently
    skipped and shipped truncated. It must instead be regenerated.
    """
    root = _partial_tree(tmp_path)
    dest = tmp_path / "bundle"
    (dest / "AIAL" / "chain_00" / "frames").mkdir(parents=True)
    (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").write_bytes(b"partial")

    made = tmp_path / "regen"
    made.mkdir()
    for i in range(5):
        (made / f"{i:05d}.jpg").write_bytes(f"regen{i}".encode())
    calls = []

    def fake_regen(state, **kw):
        calls.append(kw["neuron"])
        return made

    monkeypatch.setattr(export_bundle, "regenerate_frames", fake_regen)
    export_bundle.export_bundle(root, dest, source_tree="master", force=True)

    assert calls == ["AIAL"], "a partial frame set (frame 0 only) must trigger regeneration"
    out_frames = dest / "AIAL" / "chain_00" / "frames"
    assert sorted(p.name for p in out_frames.glob("*.jpg")) == [f"{i:05d}.jpg" for i in range(5)]
    assert (out_frames / "00000.jpg").read_bytes() == b"regen0", (
        "the stale partial frame 0 must be replaced, not left in place")


def test_slash_prefixed_recorded_path_is_never_treated_as_a_local_copy_source(tmp_path, monkeypatch):
    """Pins the copy-vs-regenerate DECISION for a /-rooted recorded frames_dir.

    ``Path("/localscratch/...").is_dir()`` is already False on Windows, so
    ``test_dead_scratch_path_triggers_regeneration`` passes even with the
    ``str(recorded).startswith("/")`` guard deleted: it never actually exercises
    the guard, it just falls through to regeneration because the path happens
    not to exist here. This test fakes ``Path.is_dir`` so the recorded slash
    path reports as a real, existing directory (what a machine where that
    absolute path genuinely resolves would see), so the guard itself, not a
    missing-directory accident, is what is left keeping regeneration the path
    taken.
    """
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"px")
    recorded = "/localscratch/4812345/frames/AIAL_chain00_s8"
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0, "frames_dir": recorded,
         "anchor_catmaid_z": 1402, "crop_window": None, "save_downscale": 8}),
        encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    real_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir",
        lambda self: True if self == Path(recorded) else real_is_dir(self))

    copy_srcs = []
    monkeypatch.setattr(export_bundle, "_replace_frames_dir",
                        lambda src, dst: copy_srcs.append(Path(src)))

    regen_calls = []

    def fake_regen(state, **kw):
        regen_calls.append(kw["neuron"])
        return tmp_path / "regen_out"

    monkeypatch.setattr(export_bundle, "regenerate_frames", fake_regen)

    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")

    assert regen_calls == ["AIAL"], "a /-rooted recorded path must be regenerated, not copied"
    assert copy_srcs == [tmp_path / "regen_out"], (
        "must copy from the regenerated dir, never from the recorded /-path")


# ---------------------------------------------------------------------------
# regenerate_frames: torch-free coverage of the tier-2 vs legacy branch and
# the alignment.CropWindow field mapping.
# ---------------------------------------------------------------------------

def _make_state(*, crop_window=None, config=None, anchor_z=1402):
    return {
        "anchor_catmaid_z": anchor_z,
        "crop_window": crop_window,
        "config": config or {},
    }


def _install_fake_pipeline(monkeypatch, calls):
    """Put a torch-free stand-in in sys.modules['pipeline'].

    ``regenerate_frames`` does ``import pipeline`` internally (deliberately, so
    importing export_bundle never pulls torch). Installing a fake module object
    under that name means the internal import resolves to this stand-in instead
    of ever triggering the real, torch-importing package.
    """
    def fake_prepare_chain_crop_frames(chain, annotate_df, cw, *, frames_root,
                                       anchor_catmaid_z, neuron, chain_idx):
        calls["crop"] = dict(chain=chain, annotate_df=annotate_df, cw=cw,
                             frames_root=frames_root, anchor_catmaid_z=anchor_catmaid_z,
                             neuron=neuron, chain_idx=chain_idx)
        return ("crop/made", {0: anchor_catmaid_z}, 0, 1)

    def fake_prepare_video_frames(chain, annotate_df, *, scale, frames_root,
                                  anchor_catmaid_z, neuron, chain_idx):
        calls["video"] = dict(chain=chain, annotate_df=annotate_df, scale=scale,
                              frames_root=frames_root, anchor_catmaid_z=anchor_catmaid_z,
                              neuron=neuron, chain_idx=chain_idx)
        return ("video/made", {0: anchor_catmaid_z}, 0, 1)

    fake_pipeline = types.SimpleNamespace(
        prepare_chain_crop_frames=fake_prepare_chain_crop_frames,
        prepare_video_frames=fake_prepare_video_frames,
    )
    monkeypatch.setitem(sys.modules, "pipeline", fake_pipeline)


def _stub_catmaid_context(monkeypatch, chain, frames_root):
    monkeypatch.setattr(
        export_bundle, "_catmaid_context",
        lambda output_root, fr: ([chain], "annotate_df_stub", "cfg_stub", frames_root))


def test_regenerate_frames_crop_window_passes_correct_crop_window_fields(tmp_path, monkeypatch):
    """A chain WITH crop_window must go through prepare_chain_crop_frames, and the
    CropWindow it receives must carry the recorded field values under their real
    names: origin_tif, size_tif, crop_scale, sam_scale (crop_scale is NOT
    ``scale``). A wrong key here would misplace every tier-2 chain's canvas.
    """
    calls = {}
    _install_fake_pipeline(monkeypatch, calls)
    chain = {"cell_name": "AIAL", "nodes": [1]}
    _stub_catmaid_context(monkeypatch, chain, tmp_path)

    state = _make_state(crop_window={
        "origin_tif": [10.0, 20.0], "size_tif": [512, 256],
        "crop_scale": 2, "sam_scale": 8,
    })

    out = export_bundle.regenerate_frames(state, neuron="AIAL", chain_idx=0,
                                          output_root=tmp_path, frames_root=tmp_path)

    assert out == Path("crop/made")
    assert "crop" in calls and "video" not in calls
    cw = calls["crop"]["cw"]
    assert cw.origin_tif == (10.0, 20.0)
    assert cw.size_tif == (512, 256)
    assert cw.crop_scale == 2
    assert cw.sam_scale == 8
    assert calls["crop"]["anchor_catmaid_z"] == 1402
    assert calls["crop"]["neuron"] == "AIAL"
    assert calls["crop"]["chain_idx"] == 0


def test_regenerate_frames_no_crop_window_uses_video_path_with_chain_scale(tmp_path, monkeypatch):
    """A chain WITHOUT crop_window must call prepare_video_frames with the scale
    recorded in the chain's OWN state.json config, not a fabricated default.
    """
    calls = {}
    _install_fake_pipeline(monkeypatch, calls)
    chain = {"cell_name": "AIAL", "nodes": [1]}
    _stub_catmaid_context(monkeypatch, chain, tmp_path)

    state = _make_state(crop_window=None, config={"scale": 4})

    out = export_bundle.regenerate_frames(state, neuron="AIAL", chain_idx=0,
                                          output_root=tmp_path, frames_root=tmp_path)

    assert out == Path("video/made")
    assert "video" in calls and "crop" not in calls
    assert calls["video"]["scale"] == 4


def test_regenerate_frames_scale_falls_back_to_default_when_unrecorded(tmp_path, monkeypatch):
    """When the chain's state.json has no recorded scale (older state.json),
    fall back to the legacy default of 8 instead of crashing on a missing key.
    """
    calls = {}
    _install_fake_pipeline(monkeypatch, calls)
    chain = {"cell_name": "AIAL", "nodes": [1]}
    _stub_catmaid_context(monkeypatch, chain, tmp_path)

    state = _make_state(crop_window=None, config={})

    export_bundle.regenerate_frames(state, neuron="AIAL", chain_idx=0,
                                    output_root=tmp_path, frames_root=tmp_path)

    assert calls["video"]["scale"] == 8


def test_relative_frames_dir_resolves_against_the_chain_dir_not_the_cwd(tmp_path, monkeypatch):
    """Re-exporting a bundle must not copy a decoy ``frames/`` out of the cwd.

    A bundle records ``frames_dir`` as the relative ``"frames"``. Export used to
    call a bare ``Path(recorded)`` for anything not starting with ``/``, which on
    a relative value resolves against the CURRENT WORKING DIRECTORY. Running the
    export from a directory that happens to hold a ``frames/`` shipped that
    unrelated directory's contents as the reviewer's canvas.
    """
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"px")
    (d / "frames").mkdir()
    (d / "frames" / "00000.jpg").write_bytes(b"real")
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0, "frames_dir": "frames",
         "anchor_catmaid_z": 1402, "crop_window": None, "save_downscale": 8}),
        encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    decoy = tmp_path / "cwd"
    (decoy / "frames").mkdir(parents=True)
    (decoy / "frames" / "00000.jpg").write_bytes(b"decoy")
    monkeypatch.chdir(decoy)

    monkeypatch.setattr(export_bundle, "regenerate_frames", lambda *a, **k: pytest.fail(
        "the chain's own relative frames dir exists; nothing should be regenerated"))

    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    shipped = (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes()
    assert shipped == b"real", "must copy the chain's own frames, not the cwd's"
