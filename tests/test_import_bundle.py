"""Import merges only reviewer-owned files, and refuses a mismatched bundle."""
import json

import pytest

import import_bundle
from sam2_utils import bundle


def _pair(tmp_path, *, bundle_mask=b"new", master_mask=b"old", neuron="AIAL", nid=42):
    master = tmp_path / "master" / neuron / "chain_00"
    (master / "masks").mkdir(parents=True)
    (master / "masks" / "mask_1402.png").write_bytes(master_mask)
    (master / "state.json").write_text(json.dumps(
        {"neuron": neuron, "chain_idx": 0, "frames_dir": "x",
         "crop_window": None, "save_downscale": 8}), encoding="utf-8")
    (master / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    bdir = tmp_path / "bundle" / neuron / "chain_00"
    (bdir / "masks").mkdir(parents=True)
    (bdir / "masks" / "mask_1402.png").write_bytes(bundle_mask)
    (bdir / "state.json").write_text(json.dumps(
        {"neuron": neuron, "chain_idx": 0, "frames_dir": "frames",
         "crop_window": None, "save_downscale": 8}), encoding="utf-8")
    (bdir / "qc.csv").write_text("z,queue\n1402,1\n", encoding="utf-8")
    (bdir / "meta.json").write_text(json.dumps({
        "schema_version": 1, "neuron_id": nid, "cell_name": neuron, "chain_idx": 0,
        "mask_space": "_sam", "mask_scale": 8, "crop_window": None,
        "z_range": [1402, 1402], "provenance": {}}), encoding="utf-8")

    data = tmp_path / "bundle" / bundle.BUNDLE_DATA_DIR
    data.mkdir(parents=True, exist_ok=True)
    (data / bundle.BUNDLE_CHAINS_NAME).write_text(
        json.dumps([{"cell_name": neuron, "nodes": [1]}]), encoding="utf-8")
    (data / bundle.BUNDLE_NODES_NAME).write_text(
        f"node_id,x,y,z,cell_name\n1,10,20,1402,{neuron}\n", encoding="utf-8")

    man = bundle.build_manifest(bundle.index_chains(tmp_path / "bundle"), source_tree="master")
    (tmp_path / "bundle" / bundle.BUNDLE_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    return tmp_path / "bundle", tmp_path / "master"


def test_import_replaces_the_master_mask(tmp_path):
    b, m = _pair(tmp_path)
    import_bundle.import_bundle(b, m)
    assert (m / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"new"


def test_import_replaces_qc(tmp_path):
    b, m = _pair(tmp_path)
    import_bundle.import_bundle(b, m)
    assert "1402,1" in (m / "AIAL" / "chain_00" / "qc.csv").read_text(encoding="utf-8")


def test_import_does_not_touch_state_json(tmp_path):
    """state.json in a bundle has a relative frames_dir; copying it would break the master."""
    b, m = _pair(tmp_path)
    import_bundle.import_bundle(b, m)
    state = json.loads((m / "AIAL" / "chain_00" / "state.json").read_text(encoding="utf-8"))
    assert state["frames_dir"] == "x"


def test_dry_run_changes_nothing(tmp_path):
    b, m = _pair(tmp_path)
    changed = import_bundle.import_bundle(b, m, dry_run=True)
    assert changed and (m / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"old"


def test_import_reports_which_chains_changed(tmp_path):
    b, m = _pair(tmp_path)
    changed = import_bundle.import_bundle(b, m)
    assert [c["chain_dir"] for c in changed] == ["AIAL/chain_00"]


def test_identical_content_is_not_reported_as_changed(tmp_path):
    b, m = _pair(tmp_path, bundle_mask=b"same", master_mask=b"same")
    (b / "AIAL" / "chain_00" / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")
    assert import_bundle.import_bundle(b, m) == []


def test_missing_chain_in_master_is_refused(tmp_path):
    b, m = _pair(tmp_path)
    import shutil
    shutil.rmtree(m / "AIAL")
    with pytest.raises(SystemExit):
        import_bundle.import_bundle(b, m)


def test_invalid_bundle_is_refused(tmp_path):
    b, m = _pair(tmp_path)
    (b / bundle.BUNDLE_MANIFEST).unlink()
    with pytest.raises(SystemExit):
        import_bundle.import_bundle(b, m)


def test_meta_identity_mismatch_is_refused(tmp_path):
    """A bundle whose meta.json disagrees with its own index must not merge.

    The refusal has to happen before anything is written. A refusal that already
    copied a mask is not a refusal, it is a partial merge of a bundle we just
    established does not belong to this tree.
    """
    b, m = _pair(tmp_path)
    meta_path = b / "AIAL" / "chain_00" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cell_name"] = "AIAR"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    master_mask = m / "AIAL" / "chain_00" / "masks" / "mask_1402.png"
    master_qc = m / "AIAL" / "chain_00" / "qc.csv"
    before_mask = master_mask.read_bytes()
    before_qc = master_qc.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        import_bundle.import_bundle(b, m)

    assert master_mask.read_bytes() == before_mask
    assert master_qc.read_text(encoding="utf-8") == before_qc
