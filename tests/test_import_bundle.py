"""Import merges only reviewer-owned files, and refuses a mismatched bundle."""
import json

import pandas as pd
import pytest

import import_bundle
from sam2_utils import bundle, labels


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


# ---------------------------------------------------------------------------
# The root ledgers: where four of review mode's ten keys write, and the only
# place a reviewer's verdicts exist.
# ---------------------------------------------------------------------------

REVIEW_HEADER = "neuron,chain_idx,review_status,reviewer,notes,updated_at"


def _write_review(root, rows):
    lines = [REVIEW_HEADER] + list(rows)
    (root / "_review.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_review(root):
    return pd.read_csv(root / "_review.csv")


def test_a_reviewers_chain_verdict_reaches_the_master(tmp_path):
    """The round trip: `a` in review mode writes only here, so only this brings it home."""
    b, m = _pair(tmp_path)
    _write_review(b, ["AIAL,0,approved,lucinda,looks right,2026-08-19T10:00:00+00:00"])

    import_bundle.import_bundle(b, m)

    got = _read_review(m)
    row = got[(got["neuron"] == "AIAL") & (got["chain_idx"] == 0)].iloc[0]
    assert row["review_status"] == "approved"
    assert row["reviewer"] == "lucinda"


def test_a_master_row_for_a_chain_not_in_the_bundle_survives(tmp_path):
    """A row merge, not a file copy: the master holds chains the bundle never had."""
    b, m = _pair(tmp_path)
    _write_review(m, [
        "AIAL,0,unreviewed,sf,,2026-08-01T00:00:00+00:00",
        "AVAL,3,corrected,sf,fixed the seed,2026-08-02T00:00:00+00:00",
    ])
    _write_review(b, ["AIAL,0,approved,lucinda,,2026-08-19T10:00:00+00:00"])

    import_bundle.import_bundle(b, m)

    got = _read_review(m)
    assert len(got) == 2
    survivor = got[(got["neuron"] == "AVAL") & (got["chain_idx"] == 3)].iloc[0]
    assert survivor["review_status"] == "corrected"
    assert survivor["reviewer"] == "sf"
    assert survivor["notes"] == "fixed the seed"
    updated = got[(got["neuron"] == "AIAL") & (got["chain_idx"] == 0)].iloc[0]
    assert updated["review_status"] == "approved", "the bundle's chain must be replaced"


def test_review_merge_is_a_no_op_without_a_bundle_ledger(tmp_path):
    """A reviewer who recorded no dispositions must not disturb the master."""
    b, m = _pair(tmp_path)
    _write_review(m, ["AVAL,3,corrected,sf,,2026-08-02T00:00:00+00:00"])
    before = (m / "_review.csv").read_text(encoding="utf-8")

    import_bundle.import_bundle(b, m)

    assert (m / "_review.csv").read_text(encoding="utf-8") == before


def test_review_merge_ignores_rows_for_chains_outside_the_bundle(tmp_path):
    """A stale ledger inside a bundle must not import chains the bundle does not hold."""
    b, m = _pair(tmp_path)
    _write_review(b, [
        "AIAL,0,approved,lucinda,,2026-08-19T10:00:00+00:00",
        "AVAR,9,rejected,lucinda,,2026-08-19T10:00:00+00:00",
    ])

    import_bundle.import_bundle(b, m)

    got = _read_review(m)
    assert set(got["neuron"]) == {"AIAL"}


def test_dry_run_reports_ledger_rows_without_writing(tmp_path):
    b, m = _pair(tmp_path)
    _write_review(b, ["AIAL,0,approved,lucinda,,2026-08-19T10:00:00+00:00"])

    import_bundle.import_bundle(b, m, dry_run=True)

    assert not (m / "_review.csv").exists()
    counts = import_bundle.merge_review_ledger(
        b, m, {("AIAL", 0)}, dry_run=True)
    assert counts == {"updated": 0, "appended": 1}


LABEL_HEADER = ",".join(labels.LABEL_COLS)


def _label_row(neuron, chain_idx, z, *, verdict, ts, reviewer="lucinda"):
    row = {c: "" for c in labels.LABEL_COLS}
    row.update(neuron=neuron, chain_idx=chain_idx, z=z, role="flagged",
               verdict=verdict, source="approve", reviewer=reviewer, ts=ts)
    return ",".join(str(row[c]) for c in labels.LABEL_COLS)


def _write_labels(root, rows):
    (root / "_labels.csv").write_text(
        "\n".join([LABEL_HEADER] + list(rows)) + "\n", encoding="utf-8")


def test_a_reviewers_frame_verdicts_reach_the_master(tmp_path):
    b, m = _pair(tmp_path)
    _write_labels(b, [
        _label_row("AIAL", 0, 1402, verdict="wrong", ts="2026-08-19T10:00:00+00:00"),
        _label_row("AIAL", 0, 1403, verdict="ok", ts="2026-08-19T10:01:00+00:00"),
    ])

    import_bundle.import_bundle(b, m)

    got = pd.read_csv(m / "_labels.csv")
    assert sorted(got["z"]) == [1402, 1403]
    assert got.set_index("z").loc[1402, "verdict"] == "wrong"


def test_labels_merge_keeps_master_rows_for_other_chains(tmp_path):
    b, m = _pair(tmp_path)
    _write_labels(m, [_label_row("AVAL", 3, 900, verdict="ok",
                                 ts="2026-08-01T00:00:00+00:00", reviewer="sf")])
    _write_labels(b, [_label_row("AIAL", 0, 1402, verdict="wrong",
                                 ts="2026-08-19T10:00:00+00:00")])

    import_bundle.import_bundle(b, m)

    got = pd.read_csv(m / "_labels.csv")
    assert len(got) == 2
    assert set(zip(got["neuron"], got["z"])) == {("AVAL", 900), ("AIAL", 1402)}


def test_labels_merge_dedupes_on_neuron_chain_z(tmp_path):
    """The store's own primary key. A later ts on the same frame wins."""
    b, m = _pair(tmp_path)
    _write_labels(m, [_label_row("AIAL", 0, 1402, verdict="ok",
                                 ts="2026-08-01T00:00:00+00:00", reviewer="sf")])
    _write_labels(b, [_label_row("AIAL", 0, 1402, verdict="wrong",
                                 ts="2026-08-19T10:00:00+00:00")])

    counts = import_bundle.merge_labels_ledger(b, m, {("AIAL", 0)})

    assert counts == {"updated": 1, "appended": 0}
    got = pd.read_csv(m / "_labels.csv")
    assert len(got) == 1, "one row per (neuron, chain_idx, z), as the store maintains"
    assert got.iloc[0]["verdict"] == "wrong"


def test_labels_merge_keeps_the_master_row_when_it_is_the_later_one(tmp_path):
    b, m = _pair(tmp_path)
    _write_labels(m, [_label_row("AIAL", 0, 1402, verdict="ok",
                                 ts="2026-08-20T00:00:00+00:00", reviewer="sf")])
    _write_labels(b, [_label_row("AIAL", 0, 1402, verdict="wrong",
                                 ts="2026-08-19T10:00:00+00:00")])

    counts = import_bundle.merge_labels_ledger(b, m, {("AIAL", 0)})

    assert counts == {"updated": 0, "appended": 0}
    assert pd.read_csv(m / "_labels.csv").iloc[0]["verdict"] == "ok"


def test_labels_merge_is_a_no_op_without_a_bundle_ledger(tmp_path):
    b, m = _pair(tmp_path)
    _write_labels(m, [_label_row("AVAL", 3, 900, verdict="ok",
                                 ts="2026-08-01T00:00:00+00:00", reviewer="sf")])
    before = (m / "_labels.csv").read_text(encoding="utf-8")

    import_bundle.import_bundle(b, m)

    assert (m / "_labels.csv").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Cross-tree merges. This repo keeps parallel trees with identical layouts, so
# every other check here passes on the wrong one.
# ---------------------------------------------------------------------------

def _parallel_trees(tmp_path):
    """A maskseed bundle plus a boxseed tree it must NOT merge into.

    The two trees have identical layouts, which is the whole problem: the bundle
    validates, its meta.json agrees with its index, and the chain exists on both
    sides, so nothing else in the import can tell them apart.
    """
    b, m = _pair(tmp_path)
    maskseed = tmp_path / "reprop_maskseed"
    m.rename(maskseed)

    boxseed = tmp_path / "reprop_boxseed"
    chain = boxseed / "AIAL" / "chain_00"
    (chain / "masks").mkdir(parents=True)
    (chain / "masks" / "mask_1402.png").write_bytes(b"boxseed-good")
    (chain / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 0, "frames_dir": "x",
         "crop_window": None, "save_downscale": 8}), encoding="utf-8")
    (chain / "qc.csv").write_text("z,queue\n1402,0\n", encoding="utf-8")

    man_path = b / bundle.BUNDLE_MANIFEST
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["source_tree"] = "reprop_maskseed"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    return b, maskseed, boxseed


def test_a_cross_tree_import_is_refused(tmp_path):
    b, _maskseed, boxseed = _parallel_trees(tmp_path)
    good = boxseed / "AIAL" / "chain_00" / "masks" / "mask_1402.png"

    with pytest.raises(SystemExit) as excinfo:
        import_bundle.import_bundle(b, boxseed)

    message = str(excinfo.value)
    assert "reprop_maskseed" in message and "reprop_boxseed" in message, (
        "the refusal has to name both trees, or it cannot be acted on")
    assert good.read_bytes() == b"boxseed-good", "nothing may be overwritten"


def test_dry_run_also_refuses_a_cross_tree_import(tmp_path):
    """A dry run that misses the one thing it would have caught is worse than useless."""
    b, _maskseed, boxseed = _parallel_trees(tmp_path)
    with pytest.raises(SystemExit):
        import_bundle.import_bundle(b, boxseed, dry_run=True)


def test_the_matching_tree_still_imports(tmp_path):
    b, maskseed, _boxseed = _parallel_trees(tmp_path)
    import_bundle.import_bundle(b, maskseed)
    assert (maskseed / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"new"


def test_allow_tree_mismatch_overrides_deliberately(tmp_path):
    b, _maskseed, boxseed = _parallel_trees(tmp_path)
    import_bundle.import_bundle(b, boxseed, allow_tree_mismatch=True)
    assert (boxseed / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"new"


def test_a_bundle_with_no_recorded_source_tree_is_refused(tmp_path):
    b, m = _pair(tmp_path)
    man_path = b / bundle.BUNDLE_MANIFEST
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man.pop("source_tree")
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit):
        import_bundle.import_bundle(b, m)


def test_the_target_tree_is_identified_by_its_directory_name(tmp_path):
    """Pins the choice: it is what export_bundle records by default."""
    assert import_bundle.target_tree_name(tmp_path / "a" / "reprop_maskseed") == "reprop_maskseed"
