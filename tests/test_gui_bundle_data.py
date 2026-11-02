"""A bundle carries its own CATMAID slice, so a reviewer's ReviewContext opens.

The real Mac scenario: `data/chains.json` and `data/aggregate_data_pv.csv` are
both gitignored, so a reviewer who clones the repo has NEITHER. Everything here
therefore points `config.CHAINS_PATH` and `config.CSV_PATH` at paths that do not
exist, which is exactly what her machine looks like, and asserts the bundle's own
`data/` is what gets read instead.
"""
import json

import pytest

import gui
from sam2_utils import bundle


def _bundle_with_data(root, *, with_data=True):
    """A minimal bundle: one chain on disk, and optionally the CATMAID slice."""
    chain_dir = root / "AIAL" / "chain_01"
    (chain_dir / "masks").mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps(
        {"neuron": "AIAL", "chain_idx": 1, "frames_dir": "frames",
         "crop_window": None}), encoding="utf-8")
    if not with_data:
        return root

    data = root / bundle.BUNDLE_DATA_DIR
    data.mkdir(parents=True)
    # AIAL keeps BOTH its chains, in source order: chain_idx is a position in
    # this list, so a missing first chain would make chain_01 resolve to the
    # wrong skeleton.
    (data / bundle.BUNDLE_CHAINS_NAME).write_text(json.dumps([
        {"cell_name": "AIAL", "nodes": [1, 2], "is_root": True},
        {"cell_name": "AIAL", "nodes": [3, 4], "is_root": False},
    ]), encoding="utf-8")
    (data / bundle.BUNDLE_NODES_NAME).write_text(
        "node_id,x,y,z,cell_name\n"
        "1,100,200,1402,AIAL\n"
        "2,110,210,1403,AIAL\n"
        "3,120,220,1404,AIAL\n"
        "4,130,230,1405,AIAL\n", encoding="utf-8")
    return root


@pytest.fixture
def _no_config_data(monkeypatch, tmp_path):
    """Point the config paths at files that do not exist, as on a review Mac."""
    monkeypatch.setattr(gui.config, "CSV_PATH", tmp_path / "nope" / "aggregate_data_pv.csv")
    monkeypatch.setattr(gui.config, "CHAINS_PATH", tmp_path / "nope" / "chains.json")


def test_find_chain_resolves_from_the_bundles_own_chains_json(tmp_path, _no_config_data):
    root = _bundle_with_data(tmp_path / "bundle")
    ctx = gui.ReviewContext(root)
    chain = ctx.find_chain("AIAL", 1)
    assert chain is not None, "the bundle's data/chains.json must be what gets read"
    assert chain["nodes"] == [3, 4], "chain_idx is a position; the list must be complete"


def test_annotate_df_is_built_from_the_bundles_own_nodes_csv(tmp_path, _no_config_data):
    root = _bundle_with_data(tmp_path / "bundle")
    ctx = gui.ReviewContext(root)
    df = ctx.annotate_df
    assert list(df["node_id"]) == [1, 2, 3, 4]
    assert {"x_tif", "y_tif"}.issubset(df.columns), (
        "the affine must be applied on read, since the bundle ships raw columns")
    assert df["x_tif"].notna().all()


def test_without_the_data_slice_the_config_fallback_is_what_fails(tmp_path, _no_config_data):
    """The bug this fixes: a bundle with no data/ raises on the first chain opened."""
    root = _bundle_with_data(tmp_path / "bundle", with_data=False)
    ctx = gui.ReviewContext(root)
    with pytest.raises(FileNotFoundError):
        ctx.find_chain("AIAL", 1)
    with pytest.raises(FileNotFoundError):
        _ = gui.ReviewContext(root).annotate_df


def test_explicit_arguments_still_win_over_the_bundle_slice(tmp_path, _no_config_data):
    """The existing seams keep their meaning: a caller-supplied list is authoritative."""
    root = _bundle_with_data(tmp_path / "bundle")
    given = [{"cell_name": "AIAL", "nodes": [9]}]
    ctx = gui.ReviewContext(root, chains=given)
    assert ctx.find_chain("AIAL", 0)["nodes"] == [9]
