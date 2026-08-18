"""Bundle logic: index, manifest, path rewrite, validation, progress."""
import json

from sam2_utils import bundle


def _chain(root, neuron, idx, *, frames="F:\\x\\frames"):
    d = root / neuron / f"chain_{idx:02d}"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"")
    (d / "state.json").write_text(json.dumps(
        {"neuron": neuron, "chain_idx": idx, "frames_dir": frames,
         "crop_window": None, "save_downscale": 8}), encoding="utf-8")
    (d / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    return d


def _data_slice(root, neurons=("AIAL",)):
    """The CATMAID slice every bundle must carry, which validate now requires."""
    d = root / bundle.BUNDLE_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / bundle.BUNDLE_CHAINS_NAME).write_text(json.dumps(
        [{"cell_name": n, "nodes": [1, 2]} for n in neurons]), encoding="utf-8")
    rows = ["node_id,x,y,z,cell_name"]
    rows += [f"{i + 1},10,20,1402,{n}" for i, n in enumerate(neurons)]
    (d / bundle.BUNDLE_NODES_NAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return d


def test_index_finds_every_chain(tmp_path):
    _chain(tmp_path, "AIAL", 0)
    _chain(tmp_path, "AIYL", 1)
    found = bundle.index_chains(tmp_path)
    assert {(c["cell_name"], c["chain_idx"]) for c in found} == {("AIAL", 0), ("AIYL", 1)}


def test_index_respects_the_neuron_filter(tmp_path):
    _chain(tmp_path, "AIAL", 0)
    _chain(tmp_path, "AIYL", 1)
    found = bundle.index_chains(tmp_path, neurons=["AIAL"])
    assert [c["cell_name"] for c in found] == ["AIAL"]


def test_rewrite_makes_frames_dir_relative():
    state = {"neuron": "AIAL", "chain_idx": 0, "frames_dir": r"F:\deep\path\frames"}
    out = bundle.rewrite_state_frames_dir(state)
    assert out["frames_dir"] == "frames"
    assert state["frames_dir"] == r"F:\deep\path\frames", "must not mutate the input"


def test_manifest_carries_schema_and_chains(tmp_path):
    _chain(tmp_path, "AIAL", 0)
    man = bundle.build_manifest(bundle.index_chains(tmp_path), source_tree="tree_a")
    assert man["schema_version"] == bundle.BUNDLE_SCHEMA_VERSION
    assert man["source_tree"] == "tree_a"
    assert len(man["chains"]) == 1


def test_validate_accepts_a_well_formed_bundle(tmp_path):
    d = _chain(tmp_path, "AIAL", 0, frames="frames")
    (d / "frames").mkdir()
    (d / "frames" / "00000.jpg").write_bytes(b"")
    (d / "meta.json").write_text(json.dumps({
        "schema_version": 1, "neuron_id": 42, "cell_name": "AIAL", "chain_idx": 0,
        "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
        "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")
    man = bundle.build_manifest(bundle.index_chains(tmp_path), source_tree="t")
    (tmp_path / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    _data_slice(tmp_path)
    assert bundle.validate_bundle(tmp_path) == []


def test_validate_reports_an_absolute_frames_dir(tmp_path):
    d = _chain(tmp_path, "AIAL", 0, frames=r"F:\still\absolute")
    (d / "meta.json").write_text(json.dumps({
        "schema_version": 1, "neuron_id": 42, "cell_name": "AIAL", "chain_idx": 0,
        "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
        "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")
    man = bundle.build_manifest(bundle.index_chains(tmp_path), source_tree="t")
    (tmp_path / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    problems = bundle.validate_bundle(tmp_path)
    assert any("absolute" in p for p in problems)


def test_validate_reports_a_missing_manifest(tmp_path):
    assert any("bundle.json" in p for p in bundle.validate_bundle(tmp_path))


def test_validate_refuses_a_newer_schema(tmp_path):
    (tmp_path / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
        {"schema_version": bundle.BUNDLE_SCHEMA_VERSION + 1, "chains": [],
         "source_tree": "t"}), encoding="utf-8")
    assert any("schema_version" in p for p in bundle.validate_bundle(tmp_path))


def test_review_progress_counts_reviewed_chains(tmp_path):
    _chain(tmp_path, "AIAL", 0)
    _chain(tmp_path, "AIAL", 1)
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        "AIAL,0,approved,lucinda,,2026-08-18T00:00:00+00:00\n", encoding="utf-8")
    prog = bundle.review_progress(tmp_path)
    assert prog["AIAL"] == {"total": 2, "reviewed": 1}


def test_reviewer_owned_is_the_documented_set():
    assert bundle.REVIEWER_OWNED == ("masks", "qc.csv")


def test_validate_reports_a_missing_data_slice(tmp_path):
    """A bundle with no data/ cannot be opened, so validation has to say so.

    Both source tables are gitignored. Without the bundle's own copies the GUI
    falls back to config.CHAINS_PATH / config.CSV_PATH, which do not exist on a
    reviewer's machine, and every chain she opens raises FileNotFoundError.
    """
    d = _chain(tmp_path, "AIAL", 0, frames="frames")
    (d / "frames").mkdir()
    (d / "meta.json").write_text(json.dumps({
        "schema_version": 1, "neuron_id": 42, "cell_name": "AIAL", "chain_idx": 0,
        "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
        "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")
    man = bundle.build_manifest(bundle.index_chains(tmp_path), source_tree="t")
    (tmp_path / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")

    problems = bundle.validate_bundle(tmp_path)
    assert any("data/chains.json" in p for p in problems)
    assert any("data/nodes.csv" in p for p in problems)


def test_validate_accepts_a_bundle_once_the_data_slice_is_there(tmp_path):
    d = _chain(tmp_path, "AIAL", 0, frames="frames")
    (d / "frames").mkdir()
    (d / "meta.json").write_text(json.dumps({
        "schema_version": 1, "neuron_id": 42, "cell_name": "AIAL", "chain_idx": 0,
        "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
        "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")
    man = bundle.build_manifest(bundle.index_chains(tmp_path), source_tree="t")
    (tmp_path / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    _data_slice(tmp_path)
    assert bundle.validate_bundle(tmp_path) == []
