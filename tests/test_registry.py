"""Registry ids are permanent. These tests are the guard on that promise."""
import csv

import pytest

from sam2_utils import registry


def _write_registry(tmp_path, rows):
    path = tmp_path / "neuron_registry.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=registry.REGISTRY_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def test_load_registry_maps_name_to_id(tmp_path):
    path = _write_registry(tmp_path, [
        {"neuron_id": 1, "cell_name": "ADAL", "first_seen": "2026-08-18", "notes": ""},
        {"neuron_id": 2, "cell_name": "AIAL", "first_seen": "2026-08-18", "notes": ""},
    ])
    assert registry.load_registry(path) == {"ADAL": 1, "AIAL": 2}


def test_lookups_round_trip(tmp_path):
    path = _write_registry(tmp_path, [
        {"neuron_id": 7, "cell_name": "AIYR", "first_seen": "2026-08-18", "notes": ""},
    ])
    reg = registry.load_registry(path)
    assert registry.neuron_id("AIYR", registry=reg) == 7
    assert registry.neuron_name(7, registry=reg) == "AIYR"


def test_unknown_name_fails_loudly(tmp_path):
    path = _write_registry(tmp_path, [
        {"neuron_id": 1, "cell_name": "ADAL", "first_seen": "2026-08-18", "notes": ""},
    ])
    reg = registry.load_registry(path)
    with pytest.raises(registry.UnknownNeuronError):
        registry.neuron_id("NOT_A_NEURON", registry=reg)
    with pytest.raises(registry.UnknownNeuronError):
        registry.neuron_name(999, registry=reg)


def test_zero_is_reserved_for_background(tmp_path):
    path = _write_registry(tmp_path, [
        {"neuron_id": 0, "cell_name": "BAD", "first_seen": "2026-08-18", "notes": ""},
    ])
    with pytest.raises(ValueError):
        registry.load_registry(path)


def test_duplicate_id_is_rejected(tmp_path):
    path = _write_registry(tmp_path, [
        {"neuron_id": 1, "cell_name": "ADAL", "first_seen": "2026-08-18", "notes": ""},
        {"neuron_id": 1, "cell_name": "AIAL", "first_seen": "2026-08-18", "notes": ""},
    ])
    with pytest.raises(ValueError):
        registry.load_registry(path)


def test_shipped_registry_is_loadable_and_stable():
    """The real registry must load, and these ids must never move."""
    reg = registry.load_registry()
    assert len(reg) > 100
    assert len(set(reg.values())) == len(reg)
    assert min(reg.values()) == 1
