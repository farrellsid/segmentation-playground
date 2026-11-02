"""_ensure_cached_frames must never leave a partially-written frame under its
real name in the shared JPEG cache, even when two different chains race to
produce the same z at once.

Before the first fix the write was `cv2.imwrite(str(cache_dir / f"z{key}.jpg"),
img)` directly: a single non-atomic write to the final name. That cache is
SHARED across every chain (keyed by z + scale, not by chain), and before
Task 3's background `_prefetch_next` thread existed, it only ever ran
serially on the GUI thread, so the lack of atomicity was never actually
reachable concurrently. Task 3 made it reachable: a prefetch thread can now
decode a z also wanted by the GUI thread (or another prefetch) on an
overlapping-z chain, and if the process dies mid `cv2.imwrite` (e.g. the
window is closed during a prefetch), the half-written file passes the
`exists()` check forever after and every later chain silently reads corrupt
data from it.

The first fix (write to a fixed `z{key}.part.jpg` sibling, then `Path.replace`
it onto the real name) closed the single-writer case but not the cross-writer
one: two different chains with overlapping z-ranges have different
`_prep_locks` keys, so this function is not serialized across them, and both
can find `z{key}.jpg` missing and start writing to the SAME `z{key}.part.jpg`
at once, exposing a partial file under the final name anyway, or making the
losing `replace` raise.

The second fix made the temp name unique PER CALL (pid + thread id), so two
concurrent writers for the same key never touch the same inode, and a losing
`replace` (both finished, both tried to claim the final name) is treated as a
benign no-op exactly when the destination already exists, since both writers
decoded the same source frame into byte-identical output. But that fix only
wrapped the RENAME in try/except: a failure inside `cv2.imwrite` itself
(raises, or returns `False`) sat outside the cleanup block, so it orphaned
its own temp file forever, and because the temp name includes the thread id,
a later retry would never reuse (and so never clean up) that name.

The third fix brings `cv2.imwrite` and the rename inside the SAME
try/except, catching broadly (not just `OSError`, since `cv2.imwrite` can
raise `cv2.error`, which is not an `OSError` subclass) so every failure
shape -- a raised exception from either step, or a silent `False` return
from `cv2.imwrite` -- routes through the same cleanup-then-maybe-reraise
logic: benign and swallowed when `dst` already exists (another writer won),
genuine and re-raised (after cleaning up the temp file) otherwise.

Torch-free, GPU-free:
    py -3 -m pytest tests/test_ensure_cached_frames_atomic.py
"""
from __future__ import annotations

import pathlib
import sys
import threading

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
    assert list(cache_dir.glob("*.part.jpg")) == [], "no temp file of any name should remain"


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
    assert list(cache_dir.glob("*.part.jpg")) == [], (
        "a write failure must not orphan its own temp file either")


def test_two_racing_writers_for_the_same_key_each_get_a_unique_temp_name(tmp_path, monkeypatch):
    """Direct proof the per-writer temp name actually closes the cross-chain race:
    two different chains with overlapping z-ranges are NOT serialized against each
    other by the per-chain `_prep_locks`, so both can see z{key}.jpg missing and
    start writing at once. Simulated here deterministically (no real-thread timing
    dependence, same philosophy as test_gui_prefetch_next.py's _SyncThread) by
    forcing two different `threading.get_ident()` values across two back-to-back
    calls for the same key, and recording the temp path each call actually wrote
    to. Before the per-call-unique fix both calls used the exact same fixed
    `z{key}.part.jpg`; this proves they now never collide, and that both still
    land the destination correctly."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"

    idents = iter([111, 222])
    monkeypatch.setattr(threading, "get_ident", lambda: next(idents))

    written = []
    real_imwrite = cv2.imwrite

    def _recording_imwrite(path, img):
        written.append(pathlib.Path(path))
        return real_imwrite(path, img)

    monkeypatch.setattr(cv2, "imwrite", _recording_imwrite)

    _ensure_cached_frames([(100, src)], cache_dir, scale=1)
    dst = cache_dir / "z100.jpg"
    # simulate the second writer's exists() check having already run before the
    # first writer finished (the actual race): force "missing" again for the
    # same key and let the second simulated thread (ident 222) write it too.
    dst.unlink()
    _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    assert len(written) == 2
    assert written[0] != written[1], "each writer must get its own temp file name"
    assert dst.exists()
    assert cv2.imread(str(dst)) is not None, "destination must be a fully decodable JPEG"
    assert list(cache_dir.glob("*.part.jpg")) == [], "no orphan temp files after the race"


def test_a_losing_rename_is_swallowed_when_another_writer_already_landed_the_file(
        tmp_path, monkeypatch):
    """Classification branch 1: tmp.replace(dst) raises, but another writer's own
    replace landed dst in the gap between this call's exists() check and its own
    replace attempt. This must be treated as a benign, redundant loss: no exception,
    own temp file cleaned up, winner's bytes left alone.

    dst must NOT exist before `_ensure_cached_frames` is called here: the function's
    own up-front `missing` filter checks `exists()` first and returns early if the
    key is already cached, which would make this test pass without ever reaching
    cv2.imwrite, the patched replace, or the except-branch at all (exactly the gap a
    prior version of this test had -- it pre-created dst, so every assertion passed
    trivially even with the benign-swallow branch deleted entirely). Instead, the
    patched `Path.replace` stand-in itself creates dst (with different, "winning"
    bytes) and THEN raises, modeling the real race: both writers' exists() checks
    both saw "missing", and the other writer's replace happened to land first, in
    the narrow window just before this writer's own replace call."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"
    cache_dir.mkdir(parents=True)
    dst = cache_dir / "z100.jpg"
    assert not dst.exists(), "dst must still be missing when _ensure_cached_frames starts"

    def _replace_that_loses_the_race(self, target):
        pathlib.Path(target).write_bytes(b"already written by the winning writer")
        raise PermissionError("simulated Windows sharing violation on the losing rename")

    monkeypatch.setattr(pathlib.Path, "replace", _replace_that_loses_the_race)

    _ensure_cached_frames([(100, src)], cache_dir, scale=1)   # must NOT raise

    assert dst.read_bytes() == b"already written by the winning writer", (
        "the winner's file must be left alone, not overwritten by the loser")
    assert list(cache_dir.glob("*.part.jpg")) == [], "the loser's own temp file must be cleaned up"


def test_an_imwrite_failure_is_also_swallowed_when_another_writer_already_landed_the_file(
        tmp_path, monkeypatch):
    """Same classification branch as above, exercised via the OTHER failure site:
    cv2.imwrite itself is what fails (a silent `False` return here) after another
    writer's full write+rename already landed dst, rather than the rename failing.
    Must still be swallowed, not re-raised, since the cache ends up correctly
    populated by the winner either way."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"
    cache_dir.mkdir(parents=True)
    dst = cache_dir / "z100.jpg"
    assert not dst.exists()

    def _imwrite_that_loses_the_race(path, img):
        dst.write_bytes(b"already written by the winning writer")
        return False   # silent encode failure -> _ensure_cached_frames raises IOError internally

    monkeypatch.setattr(cv2, "imwrite", _imwrite_that_loses_the_race)

    _ensure_cached_frames([(100, src)], cache_dir, scale=1)   # must NOT raise

    assert dst.read_bytes() == b"already written by the winning writer"
    assert list(cache_dir.glob("*.part.jpg")) == []


def test_a_losing_rename_still_raises_when_the_destination_was_never_written(
        tmp_path, monkeypatch):
    """Classification branch 2: tmp.replace(dst) raises and dst does NOT exist. This is
    a genuine, unrelated failure (no other writer actually won), so it must propagate
    rather than being swallowed, with the caller's own temp file still cleaned up."""
    src = tmp_path / "src" / "z100.tif"
    _write_fixture_image(src)
    cache_dir = tmp_path / "frames_cache_s1"

    def _failing_replace(self, target):
        raise PermissionError("simulated genuine rename failure, no winner exists")

    monkeypatch.setattr(pathlib.Path, "replace", _failing_replace)

    with pytest.raises(PermissionError):
        _ensure_cached_frames([(100, src)], cache_dir, scale=1)

    dst = cache_dir / "z100.jpg"
    assert not dst.exists()
    assert list(cache_dir.glob("*.part.jpg")) == [], "a genuine failure must not orphan the temp file"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
