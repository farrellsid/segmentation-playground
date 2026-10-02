"""Idempotent `prepare_chain_crop_frames`: a second call with identical inputs
must be a pure cache hit (zero cv2.imwrite calls), and a changed z_range or
crop window must still rebuild. See
docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md.

Torch-free, GPU-free:
    py -3 -m pytest tests/test_prepare_chain_crop_frames_cache.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from pipeline import crop
from sam2_utils.alignment import CropWindow


class _FakeFrameStore:
    """Identity z<->key mapping. The paths this returns are never actually
    read: ``_read_tif_window`` is monkeypatched below to hand back a constant
    frame, so this test exercises the caching logic without real tif fixtures."""

    def key_of_z(self, z):
        return int(z)

    def z_of_key(self, key):
        return int(key)

    def files_in_z_range(self, z0, z1):
        lo, hi = sorted((int(z0), int(z1)))
        return [(z, pathlib.Path(f"z{z}.tif")) for z in range(lo, hi + 1)]


@pytest.fixture(autouse=True)
def _fake_tif_reads(monkeypatch):
    monkeypatch.setattr(crop, "_read_tif_window",
                        lambda src_path, sl: np.zeros((10, 10, 3), dtype=np.uint8))


def _prepare(tmp_path, *, cw=None, anchor_catmaid_z=102, z_range=(100, 104)):
    cw = cw or CropWindow(origin_tif=(0.0, 0.0), size_tif=(10, 10),
                          crop_scale=1, sam_scale=8)
    return crop.prepare_chain_crop_frames(
        chain={}, annotate_df=None, cw=cw, frames_root=tmp_path,
        anchor_catmaid_z=anchor_catmaid_z, neuron="TESTN", chain_idx=0,
        frame_store=_FakeFrameStore(), z_range=z_range)


def _count_imwrite(monkeypatch):
    calls = {"n": 0}
    real_imwrite = cv2.imwrite

    def _counting_imwrite(*args, **kwargs):
        calls["n"] += 1
        return real_imwrite(*args, **kwargs)

    monkeypatch.setattr(cv2, "imwrite", _counting_imwrite)
    return calls


def _count_files_in_z_range(monkeypatch):
    calls = {"n": 0}
    orig_method = _FakeFrameStore.files_in_z_range

    def _counting_files_in_z_range(self, z0, z1):
        calls["n"] += 1
        return orig_method(self, z0, z1)

    monkeypatch.setattr(_FakeFrameStore, "files_in_z_range", _counting_files_in_z_range)
    return calls


def test_second_call_with_identical_inputs_makes_no_imwrite_calls(tmp_path, monkeypatch):
    first = _prepare(tmp_path)

    calls_imwrite = _count_imwrite(monkeypatch)
    calls_glob = _count_files_in_z_range(monkeypatch)
    second = _prepare(tmp_path)

    assert calls_imwrite["n"] == 0, "a matching second call must be a pure cache hit: zero imwrite calls"
    assert calls_glob["n"] == 0, "a matching second call must skip the glob: zero files_in_z_range calls"
    assert second == first


def test_a_changed_z_range_invalidates_the_cache_and_rebuilds(tmp_path, monkeypatch):
    _prepare(tmp_path, z_range=(100, 104))   # 5 frames

    calls = _count_imwrite(monkeypatch)
    second = _prepare(tmp_path, z_range=(100, 105))   # 6 frames, one more

    assert calls["n"] == 6, "a wider z_range must rebuild, one imwrite per frame"
    assert second[3] == 6


def test_a_changed_crop_window_invalidates_the_cache_and_rebuilds(tmp_path, monkeypatch):
    _prepare(tmp_path)   # default crop_scale=1 window at origin (0, 0)

    calls = _count_imwrite(monkeypatch)
    other_cw = CropWindow(origin_tif=(5.0, 5.0), size_tif=(10, 10),
                          crop_scale=1, sam_scale=8)
    _prepare(tmp_path, cw=other_cw)   # same view_dir name (same crop_scale), different window

    assert calls["n"] == 5, "a different crop window must rebuild, not reuse the old view"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
