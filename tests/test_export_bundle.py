"""Export produces a bundle that validates and carries relative frame paths."""
import json
import sys
import types
from pathlib import Path

import pytest

import export_bundle
from sam2_utils import bundle

#: Captured before the autouse stub replaces the module attribute, so the tests
#: that exercise regeneration itself still reach the real implementation.
_REAL_REGENERATE_FRAMES = export_bundle.regenerate_frames


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
             "anchor_catmaid_z": 1402, "n_frames": 1,
             "crop_window": {"origin_tif": [0.0, 0.0], "size_tif": [512, 512],
                             "crop_scale": 2, "sam_scale": 8},
             "save_downscale": 8}), encoding="utf-8")
        (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch):
    monkeypatch.setattr(export_bundle.registry, "load_registry",
                        lambda *a, **k: {"AIAL": 42, "AIYL": 7})


@pytest.fixture(autouse=True)
def _stub_regenerate(monkeypatch, tmp_path):
    """Stand in for frame regeneration, which every export now performs.

    Export never copies a recorded frames_dir, so the real
    ``regenerate_frames`` is on the path of every test here. It reaches
    ``pipeline`` and the raw EM store, so stubbing it is what keeps this suite
    torch-free and runnable with no data. A test that cares about regeneration
    overrides this with its own stub.
    """
    made = tmp_path / "_stub_frames"
    made.mkdir(exist_ok=True)
    (made / "00000.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(export_bundle, "regenerate_frames", lambda state, **kw: made)


#: A stand-in for data/chains.json: three neurons, and AIAL deliberately has TWO
#: chains so a filtered export can be checked for dropping one of them.
_SOURCE_CHAINS = [
    {"cell_name": "AIAL", "nodes": [1, 2], "is_root": True},
    {"cell_name": "AIYL", "nodes": [3], "is_root": True},
    {"cell_name": "AIAL", "nodes": [4, 5], "is_root": False},
    {"cell_name": "AVAL", "nodes": [6], "is_root": True},
]


def _source_nodes():
    import pandas as pd
    df = pd.DataFrame({
        "node_id": [1, 2, 3, 4, 5, 6],
        "x": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "y": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        "z": [1402, 1403, 1402, 1404, 1405, 1402],
        "cell_name": ["AIAL", "AIAL", "AIYL", "AIAL", "AIAL", "AVAL"],
    })
    # _catmaid_context adds these before anyone downstream sees the frame; the
    # bundle must ship the RAW columns, so the writer has to strip them again.
    df["x_tif"], df["y_tif"] = df["x"] * 2, df["y"] * 2
    return df


@pytest.fixture(autouse=True)
def _fake_source_data(monkeypatch, tmp_path):
    """Stand in for the CATMAID tables the export reads its data/ slice from.

    The real ones live on paths this machine may not have. Stubbing the context
    (rather than skipping the slice) means every export test still exercises the
    data/ write, which is what makes a bundle openable on a reviewer's machine.
    """
    monkeypatch.setattr(export_bundle, "_check_source_data_present", lambda: None)
    monkeypatch.setattr(
        export_bundle, "_catmaid_context",
        lambda output_root, frames_root: (_SOURCE_CHAINS, _source_nodes(), "cfg", tmp_path))


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

    out = _REAL_REGENERATE_FRAMES(state, neuron="AIAL", chain_idx=0,
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

    out = _REAL_REGENERATE_FRAMES(state, neuron="AIAL", chain_idx=0,
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

    _REAL_REGENERATE_FRAMES(state, neuron="AIAL", chain_idx=0,
                                    output_root=tmp_path, frames_root=tmp_path)

    assert calls["video"]["scale"] == 8


def test_no_local_frames_dir_is_ever_copied_however_plausible(tmp_path, monkeypatch):
    """Export regenerates unconditionally, so no on-disk frames dir is trusted.

    Two independent real failures made copying untenable. The recorded path is
    usually a dead Narval scratch dir. And when it does exist, a
    `gui.py --anchor-only --context-frames N` review session will have rewritten
    it to hold only the anchor plus or minus N: the first real export found all
    18 AIYL chains left with exactly 5 frames, for chains needing 1 to 43.

    This test sets up the most plausible-looking copy source there is, the
    chain's own frames dir, present and non-empty, with a decoy of the same name
    in the cwd for good measure, and asserts neither is shipped.
    """
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"px")
    (d / "frames").mkdir()
    (d / "frames" / "00000.jpg").write_bytes(b"stale-local")
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0, "frames_dir": "frames",
         "anchor_catmaid_z": 1402, "n_frames": 1,
         "crop_window": None, "save_downscale": 8}),
        encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    decoy = tmp_path / "cwd"
    (decoy / "frames").mkdir(parents=True)
    (decoy / "frames" / "00000.jpg").write_bytes(b"decoy")
    monkeypatch.chdir(decoy)

    regenerated = tmp_path / "regen"
    regenerated.mkdir()
    (regenerated / "00000.jpg").write_bytes(b"regenerated")
    monkeypatch.setattr(export_bundle, "regenerate_frames", lambda *a, **k: regenerated)

    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    shipped = (dest / "AIAL" / "chain_00" / "frames" / "00000.jpg").read_bytes()
    assert shipped == b"regenerated", f"copied something instead of regenerating: {shipped!r}"


# ---------------------------------------------------------------------------
# The data/ slice: what makes a bundle openable on a machine with no data dir.
# ---------------------------------------------------------------------------

def test_export_writes_the_catmaid_slice(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    assert (dest / "data" / "chains.json").exists()
    assert (dest / "data" / "nodes.csv").exists()


def test_data_slice_keeps_every_chain_of_an_exported_neuron(tmp_path):
    """Filter by NEURON, never by chain.

    ``chain_idx`` is a POSITION in a neuron's chain list. Dropping AIAL's second
    chain because it was not the one exported would renumber it, and the GUI
    would open the wrong chain for every index after the gap.
    """
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, neurons=["AIAL"], source_tree="master")
    shipped = json.loads((dest / "data" / "chains.json").read_text(encoding="utf-8"))
    assert [c["cell_name"] for c in shipped] == ["AIAL", "AIAL"]
    assert [c["nodes"] for c in shipped] == [[1, 2], [4, 5]], "order must be the source order"


def test_data_slice_drops_unexported_neurons(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, neurons=["AIAL"], source_tree="master")
    shipped = json.loads((dest / "data" / "chains.json").read_text(encoding="utf-8"))
    assert all(c["cell_name"] == "AIAL" for c in shipped)
    nodes = (dest / "data" / "nodes.csv").read_text(encoding="utf-8")
    assert "AVAL" not in nodes and "AIYL" not in nodes


def test_nodes_csv_ships_raw_columns_only(tmp_path):
    """No precomputed x_tif/y_tif: the affine stays in code, not in shipped data."""
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")
    header = (dest / "data" / "nodes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "x_tif" not in header and "y_tif" not in header
    assert header.split(",")[:2] == ["node_id", "x"]


def test_export_fails_loudly_when_the_source_tables_are_missing(tmp_path, monkeypatch):
    """A bundle with no data/ cannot be opened, so it must not be written at all."""
    root = _tree(tmp_path)

    def boom():
        raise SystemExit("missing data/chains.json")

    monkeypatch.setattr(export_bundle, "_check_source_data_present", boom)
    with pytest.raises(SystemExit):
        export_bundle.export_bundle(root, tmp_path / "bundle", source_tree="master")


def test_an_incomplete_recorded_frames_dir_is_regenerated_not_copied(tmp_path, monkeypatch):
    """A recorded frames_dir that EXISTS but is short must not be trusted.

    Found on the first real export. A `gui.py --anchor-only --context-frames 2`
    review session regenerates a chain's frames directory holding just the anchor
    plus or minus two, so every reviewed chain's recorded directory was left with
    exactly 5 frames regardless of how many the chain actually has (1 to 43 across
    the real AIYL tree). Copying it because it merely exists ships a bundle whose
    chains have a fraction of their EM, which a reviewer cannot work from.
    """
    root = tmp_path / "master"
    d = root / "AIAL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    for z in range(1402, 1412):
        (d / "masks" / f"mask_{z:04d}.png").write_bytes(b"px")
    stale = tmp_path / "stale_frames"
    stale.mkdir()
    for i in range(5):                      # 5 frames on disk, 10 the chain needs
        (stale / f"{i:05d}.jpg").write_bytes(b"anchor-only leftover")
    (d / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0, "frames_dir": str(stale),
         "anchor_catmaid_z": 1402, "n_frames": 10,
         "crop_window": None, "save_downscale": 8}), encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    regenerated = tmp_path / "regen"
    regenerated.mkdir()
    for i in range(10):
        (regenerated / f"{i:05d}.jpg").write_bytes(b"regenerated")
    calls = []

    def fake_regen(state, **kw):
        calls.append(kw["neuron"])
        return regenerated

    monkeypatch.setattr(export_bundle, "regenerate_frames", fake_regen)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master")

    assert calls == ["AIAL"], "an incomplete source must trigger regeneration"
    out = dest / "AIAL" / "chain_00" / "frames"
    assert len(list(out.glob("*.jpg"))) == 10
    assert (out / "00000.jpg").read_bytes() == b"regenerated"
