"""_ensure_cached_frames must never leave a partially-written frame under its
real name in the shared JPEG cache.

Before this fix the write was `cv2.imwrite(str(cache_dir / f"z{key}.jpg"), img)`
directly: a single non-atomic write to the final name. That cache is SHARED
across every chain (keyed by z + scale, not by chain), and before Task 3's
background `_prefetch_next` thread existed, it only ever ran serially on the
GUI thread, so the lack of atomicity was never actually reachable concurrently.
Task 3 made it reachable: a prefetch thread can now decode a z also wanted by
the GUI thread (or another prefetch) on an overlapping-z chain, and if the
process dies mid `cv2.imwrite` (e.g. the window is closed during a prefetch),
the half-written file passes the `exists()` check forever after and every
later chain silently reads corrupt data from it.

The fix: write to a `z{key}.part.jpg` sibling, then `Path.replace` it onto the
real name (the same temp-then-rename pattern `prepare_chain_crop_frames`'s
sidecar already uses; the tmp name keeps a `.jpg` suffix because cv2.imwrite
picks its codec from the extension and does not recognize `.jpg.tmp`), so a
reader only ever observes either no file or a fully-written one, and a crash
mid-write leaves no `z{key}.jpg` at all (just an orphaned `.part.jpg`).

Torch-free, GPU-free:
    py -3 -m pytest tests/test_ensure_cached_frames_atomic.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from pipeline.frames import _ensure_cached_frames


def _write_fixture_image(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((10, 10, 3), 128, dtype=np.uint8))


def test_destination_is_fully_written_and_no_tmp_file_remains(tmp_path):
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"

    _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    dst = cache_dir / "z100.jpg"
    assert dst.exists()
    assert cv2.imread(str(dst)) is not None, "destination must be a fully decodable JPEG"
    assert not (cache_dir / "z100.part.jpg").exists()


def test_second_call_is_a_noop_cache_hit(tmp_path, monkeypatch):
    """Unchanged pre-existing behavior: once z{key}.jpg exists, a later call
    must not re-decode the source (no imread of src_path)."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"
    _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    calls = []
    real_imread = cv2.imread

    def _counting_imread(path, *a, **k):
        calls.append(path)
        return real_imread(path, *a, **k)

    monkeypatch.setattr(cv2, "imread", _counting_imread)
    _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    assert calls == [], "a cached z must not be re-decoded"


def test_a_failed_write_never_produces_a_partial_destination_file(tmp_path, monkeypatch):
    """Direct proof of atomicity: if cv2.imwrite dies partway (simulated here by
    writing a sentinel to the passed path and then raising), the destination
    name must never exist afterward, since the rename that would expose it
    under its real name never happens."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"

    def _dying_imwrite(path, img):
        pathlib.Path(path).write_bytes(b"PARTIAL")   # the write that got interrupted
        raise RuntimeError("simulated crash mid cv2.imwrite")

    monkeypatch.setattr(cv2, "imwrite", _dying_imwrite)

    with pytest.raises(RuntimeError):
        _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    dst = cache_dir / "z100.jpg"
    assert not dst.exists(), "a crash mid-write must not leave a file under the real name"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
