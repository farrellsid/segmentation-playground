"""meta.json is the portable identity+geometry record exporters read."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sam2_utils import chain_meta


def _legacy_state():
    return {"neuron": "AIAL", "chain_idx": 0, "frames_dir": r"F:\x\frames",
            "crop_window": None, "save_downscale": 8}


def _tier2_state():
    return {"neuron": "AIYL", "chain_idx": 3, "frames_dir": r"F:\x\crop",
            "crop_window": {"origin_tif": [4096.0, 3712.0], "size_tif": [2048, 1792],
                            "crop_scale": 2, "sam_scale": 8},
            "save_downscale": 8}


def test_legacy_chain_records_save_downscale():
    meta = chain_meta.build_meta(_legacy_state(), neuron_id=42, source_tree="tree_a")
    assert meta["mask_space"] == "_sam"
    assert meta["mask_scale"] == 8
    assert meta["crop_window"] is None
    assert meta["neuron_id"] == 42
    assert meta["cell_name"] == "AIAL"
    assert meta["schema_version"] == chain_meta.META_SCHEMA_VERSION


def test_tier2_chain_records_crop_scale_not_save_downscale():
    """The trap: assuming one global mask scale misplaces every tier-2 chain."""
    meta = chain_meta.build_meta(_tier2_state(), neuron_id=7, source_tree="tree_b")
    assert meta["mask_space"] == "_pcrop"
    assert meta["mask_scale"] == 2
    assert meta["crop_window"]["origin_tif"] == [4096.0, 3712.0]


def test_z_range_comes_from_the_mask_files(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    for z in (1402, 1403, 1490):
        (masks / f"mask_{z:04d}.png").write_bytes(b"")
    meta = chain_meta.build_meta(_legacy_state(), neuron_id=1, source_tree="t",
                                 chain_dir=tmp_path)
    assert meta["z_range"] == [1402, 1490]


def test_write_then_read_round_trips(tmp_path):
    meta = chain_meta.build_meta(_legacy_state(), neuron_id=42, source_tree="t")
    chain_meta.write_meta(tmp_path, meta)
    assert chain_meta.read_meta(tmp_path) == meta


def test_validate_rejects_missing_field():
    meta = chain_meta.build_meta(_legacy_state(), neuron_id=42, source_tree="t")
    del meta["neuron_id"]
    with pytest.raises(ValueError):
        chain_meta.validate_meta(meta)


def test_validate_rejects_future_schema_version():
    meta = chain_meta.build_meta(_legacy_state(), neuron_id=42, source_tree="t")
    meta["schema_version"] = chain_meta.META_SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        chain_meta.validate_meta(meta)


def test_module_does_not_import_pipeline():
    """The whole point: an exporter can read meta.json without the pipeline.

    Runs in a fresh subprocess, mirroring ``test_import_stays_light`` in
    ``tests/test_registry.py``. An in-process check is unreliable here: the
    pytest session may have already imported ``pipeline`` for other tests,
    and reloading ``chain_meta`` afterward would not undo that, making the
    check vacuous regardless of what ``chain_meta`` itself imports.
    """
    code = (
        "import sys\n"
        "from sam2_utils import chain_meta\n"
        "offenders = [m for m in sys.modules "
        "if m == 'pipeline' or m.startswith('pipeline.')]\n"
        "print('OFFENDERS:' + ','.join(offenders))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"subprocess import of sam2_utils.chain_meta failed:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    offender_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("OFFENDERS:")),
        "OFFENDERS:<missing>",
    )
    offenders = [m for m in offender_line[len("OFFENDERS:"):].split(",") if m]
    assert offenders == [], (
        f"from sam2_utils import chain_meta pulled in pipeline: {offenders}\n"
        f"stdout={proc.stdout}"
    )


def test_meta_is_plain_json(tmp_path):
    meta = chain_meta.build_meta(_tier2_state(), neuron_id=7, source_tree="t")
    chain_meta.write_meta(tmp_path, meta)
    raw = json.loads((tmp_path / chain_meta.META_FILENAME).read_text(encoding="utf-8"))
    assert raw["cell_name"] == "AIYL"
