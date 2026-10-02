"""_load_matching_prep must degrade to a cache miss (None) on a sidecar that
parses as valid JSON but is missing an expected key, or has one in the wrong
shape, rather than raising KeyError/AttributeError/TypeError out to the
caller.

prepare_chain_crop_frames (which calls _load_matching_prep to decide whether
to reuse a view directory) also runs in the batch/orchestrator path on
Narval, not just interactively from the GUI: a malformed sidecar there must
trigger a rebuild of that one chain, not abort the whole batch job.

See docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md.

Torch-free, GPU-free:
    py -3 -m pytest tests/test_load_matching_prep_malformed_sidecar.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pipeline import crop

_WINDOW = {"origin_tif": [0.0, 0.0], "size_tif": [10, 10], "crop_scale": 1, "sam_scale": 8}


def _write_sidecar(view_dir: pathlib.Path, meta: dict) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / crop._PREP_META_NAME).write_text(json.dumps(meta))


def test_a_sidecar_missing_n_frames_is_a_cache_miss_not_a_crash(tmp_path):
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        # n_frames missing entirely
        "anchor_frame_idx": 2, "frame_to_z": {"0": 100, "1": 101, "2": 102},
    })
    assert crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102) is None


def test_a_sidecar_missing_anchor_frame_idx_is_a_cache_miss_not_a_crash(tmp_path):
    # n_frames=0 so the "every frame file exists" check is vacuously true (all()
    # over an empty range), and the function actually reaches the missing key
    # below instead of returning None earlier for an unrelated reason.
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        "n_frames": 0, "frame_to_z": {},
        # anchor_frame_idx missing entirely
    })
    assert crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102) is None


def test_a_sidecar_missing_frame_to_z_is_a_cache_miss_not_a_crash(tmp_path):
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        "n_frames": 0, "anchor_frame_idx": 0,
        # frame_to_z missing entirely
    })
    assert crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102) is None


def test_a_sidecar_with_frame_to_z_in_the_wrong_shape_is_a_cache_miss_not_a_crash(tmp_path):
    # frame_to_z present but a list, not a dict: calling .items() on it raises
    # AttributeError, not KeyError or TypeError, so this specifically proves the
    # except clause is broad enough for a wrong-SHAPE value, not just a missing key.
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        "n_frames": 0, "anchor_frame_idx": 0,
        "frame_to_z": [0, 100, 1, 101],
    })
    assert crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102) is None


def test_a_well_formed_sidecar_with_no_frame_files_is_still_a_plain_cache_miss(tmp_path):
    """Sanity check the new except clause doesn't accidentally mask the ORIGINAL
    incomplete-directory guard: a well-formed sidecar claiming n_frames=3 but with
    no actual jpg files on disk must still be None, via the pre-existing path, not
    the new except clause (nothing here should raise in the first place)."""
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        "n_frames": 3, "anchor_frame_idx": 0,
        "frame_to_z": {"0": 100, "1": 101, "2": 102},
    })
    assert crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102) is None


def test_a_fully_well_formed_matching_sidecar_is_still_a_cache_hit(tmp_path):
    """The fix must not turn EVERY sidecar into a miss: a well-formed, matching
    sidecar with all its frame files actually present is still a hit, unchanged."""
    _write_sidecar(tmp_path, {
        "z_range": [100, 104], "window": _WINDOW, "anchor_catmaid_z": 102,
        "n_frames": 2, "anchor_frame_idx": 1,
        "frame_to_z": {"0": 100, "1": 102},
    })
    (tmp_path / "00000.jpg").write_bytes(b"fake-jpeg")
    (tmp_path / "00001.jpg").write_bytes(b"fake-jpeg")

    result = crop._load_matching_prep(
        tmp_path, z_range=(100, 104), window=_WINDOW, anchor_catmaid_z=102)

    assert result == (str(tmp_path), {0: 100, 1: 102}, 1, 2)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
