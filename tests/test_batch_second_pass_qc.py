"""Unit tests for batch._apply_second_pass_and_update_qc: the read/mutate/write
logic that folds pipeline.propagate.apply_second_pass's {z: "corrected"|
"guard_fallback"} outcomes back into a chain's on-disk qc.csv.

These stub out merge_metric.nodes_by_z, merge_metric.score_chain, and
apply_second_pass itself, so the test never touches a real image predictor or
CATMAID data. It exercises real file I/O against a qc.csv written the same
way the pipeline writes one (df.set_index("z").to_csv(...)).
"""
import importlib

import pandas as pd

import batch
from eval import merge_metric
from pipeline import config as cfgmod
from pipeline.state import ChainState

# pipeline/__init__.py re-exports the `propagate` *function* as the package
# attribute `pipeline.propagate`, shadowing the submodule of the same name.
# `from pipeline.propagate import apply_second_pass` (as batch.py does)
# still resolves to the real submodule via sys.modules, so fetch it the same
# way here rather than `from pipeline import propagate`, which would hand
# back the function instead of the module.
propagate = importlib.import_module("pipeline.propagate")


def _write_qc_csv(path, rows):
    pd.DataFrame(rows).set_index("z").to_csv(path)


class _StubSession:
    annotate_df = pd.DataFrame(
        {"node_id": [], "cell_name": [], "z": [], "x_tif": [], "y_tif": []})
    image_predictor = None


def test_corrected_clears_flags_and_guard_fallback_sets_them(tmp_path, monkeypatch):
    chain_dir = tmp_path / "chain_00"
    chain_dir.mkdir()
    qc_path = chain_dir / "qc.csv"
    _write_qc_csv(qc_path, {
        "z": [100, 101, 102],
        "area": [50.0, 52.0, 51.0],
        "flag": [True, True, False],
        "intervene": [True, True, False],
        "queue": [True, True, False],
    })

    monkeypatch.setattr(merge_metric, "nodes_by_z", lambda annotate_df, scale: {})
    monkeypatch.setattr(
        merge_metric, "score_chain",
        lambda chain_dir, neuron, nodes_by_z, radius: [])
    monkeypatch.setattr(
        propagate, "apply_second_pass",
        lambda *a, **k: {100: "corrected", 101: "guard_fallback"})

    cfg = cfgmod.PipelineConfig(second_pass=True)
    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    state = ChainState(
        neuron="AVAL", chain_idx=0, crop_window=None,
        frames_dir=str(tmp_path / "frames"), frame_to_z={})

    batch._apply_second_pass_and_update_qc(
        _StubSession(), cfg, "AVAL", chain, chain_dir, state)

    out = pd.read_csv(qc_path)

    corrected = out[out["z"] == 100].iloc[0]
    assert corrected["second_pass"] == "corrected"
    assert not corrected["flag"]
    assert not corrected["intervene"]
    assert not corrected["queue"]

    guard_fallback = out[out["z"] == 101].iloc[0]
    assert guard_fallback["second_pass"] == "guard_fallback"
    assert guard_fallback["flag"]
    assert guard_fallback["intervene"]
    assert guard_fallback["queue"]

    # z=102 wasn't in the outcomes dict, so its pre-existing (already-clean)
    # flags are left untouched and it never picks up a second_pass tag.
    untouched = out[out["z"] == 102].iloc[0]
    assert untouched["second_pass"] != "corrected"
    assert untouched["second_pass"] != "guard_fallback"
    assert not untouched["flag"]
    assert not untouched["intervene"]
    assert not untouched["queue"]
