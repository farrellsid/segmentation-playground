"""Backfill writes meta.json into trees that predate it (the corrected AIA/AIY trees)."""
import json

import pytest

import backfill_meta
from sam2_utils import chain_meta


def _make_chain(root, neuron, idx, state):
    d = root / neuron / f"chain_{idx:02d}"
    (d / "masks").mkdir(parents=True)
    (d / "masks" / "mask_1402.png").write_bytes(b"")
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def test_backfill_writes_one_meta_per_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill_meta.registry, "load_registry", lambda *a, **k: {"AIAL": 42})
    _make_chain(tmp_path, "AIAL", 0, {"neuron": "AIAL", "chain_idx": 0,
                                      "crop_window": None, "save_downscale": 8})
    written = backfill_meta.backfill_tree(tmp_path, source_tree="t")
    assert len(written) == 1
    meta = chain_meta.read_meta(tmp_path / "AIAL" / "chain_00")
    assert meta["neuron_id"] == 42
    assert meta["z_range"] == [1402, 1402]


def test_backfill_skips_existing_unless_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill_meta.registry, "load_registry", lambda *a, **k: {"AIAL": 42})
    d = _make_chain(tmp_path, "AIAL", 0, {"neuron": "AIAL", "chain_idx": 0,
                                          "crop_window": None, "save_downscale": 8})
    backfill_meta.backfill_tree(tmp_path, source_tree="t")
    assert backfill_meta.backfill_tree(tmp_path, source_tree="t") == []
    assert len(backfill_meta.backfill_tree(tmp_path, source_tree="t", overwrite=True)) == 1
    assert (d / "meta.json").exists()


def test_unregistered_neuron_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill_meta.registry, "load_registry", lambda *a, **k: {})
    _make_chain(tmp_path, "MYSTERY", 0, {"neuron": "MYSTERY", "chain_idx": 0,
                                         "crop_window": None, "save_downscale": 8})
    with pytest.raises(backfill_meta.registry.UnknownNeuronError):
        backfill_meta.backfill_tree(tmp_path, source_tree="t")
