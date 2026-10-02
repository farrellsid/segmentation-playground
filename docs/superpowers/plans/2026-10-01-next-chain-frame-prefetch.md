# Next-chain frame prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opening the next chain in the review GUI should feel instant, by making the next chain's frame crop happen in the background while you are still correcting the current one.

**Architecture:** Two independent pieces. (1) `pipeline/crop.py`'s `prepare_chain_crop_frames` gets a sidecar-file cache so a second call with identical inputs is a no-op instead of an unconditional rebuild. (2) `gui.py`'s `ReviewGUI.open_chain` fires a background daemon thread, once it finishes loading a chain, that runs the same frame-resolution path for whichever chain `next CHAIN` would open next; a per-chain lock keeps that thread and a later real `open_chain` call from writing the same view directory at once.

**Tech Stack:** Python 3.10+ stdlib `threading` and `json`, existing `pipeline`/`sam2_utils` modules, `pytest` (CPU-only, no torch/napari in tests).

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (repo-wide rule, `CLAUDE.md`).
- Run the `humanizer` skill on any committed prose (docs, CHANGELOG entries, commit messages) before committing it.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. New tests in this plan must not import torch or napari.
- Lint only the files touched: `ruff check <file>`, never a whole-tree `ruff format`.
- No new CLI flag: prefetch is always on, with a safe, silent fallback.
- Prefetch depth is always 1 (just the immediately-next chain), re-armed on every chain open. No backward prefetch, no multi-chain lookahead.
- The per-chain lock is keyed by `(neuron, chain_idx)` alone, not also by scale: within one `ReviewGUI` session, `self.anchor_only`/`self.context_frames`/`self.ctx.cfg` never change between chains, so a chain's view directory path is already fully determined by `(neuron, chain_idx)` for the lifetime of that session.
- Commit incrementally, one concern per task, so any step can be reverted independently.

Spec: [docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md](../specs/2026-10-01-next-chain-frame-prefetch-design.md).

---

### Task 1: Idempotent `prepare_chain_crop_frames`

**Files:**
- Modify: `pipeline/crop.py:393-462` (the `prepare_chain_crop_frames` function and its docstring)
- Modify: `pipeline/crop.py` (new helpers, inserted directly above `prepare_chain_crop_frames`, i.e. after line 390's `return out`)
- Modify: `docs/reference/state-and-storage.md:44-57` (the frames-tree diagram)
- Create: `tests/test_prepare_chain_crop_frames_cache.py`

**Interfaces:**
- Consumes: `pipeline.crop.TifFrameStore`/`FrameStore` (existing, unchanged), `sam2_utils.alignment.CropWindow.to_dict()`/`.from_dict()` (existing, unchanged), `pipeline.crop._read_tif_window`/`_downscale_image` (existing, unchanged).
- Produces: `prepare_chain_crop_frames(...)` keeps its existing signature and return type `tuple[str, dict[int, int], int, int]` exactly. Two new module-private helpers in `pipeline/crop.py`: `_load_matching_prep(view_dir, *, z_range, window, anchor_catmaid_z) -> Optional[tuple[str, dict[int, int], int, int]]` and `_write_prep_meta(view_dir, *, z_range, window, anchor_catmaid_z, n_frames, anchor_frame_idx, frame_to_z) -> None`. Task 2 and Task 3 do not call these directly; they only rely on `prepare_chain_crop_frames`'s behavior being idempotent now.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_chain_crop_frames_cache.py`:

```python
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


def test_second_call_with_identical_inputs_makes_no_imwrite_calls(tmp_path, monkeypatch):
    first = _prepare(tmp_path)

    calls = _count_imwrite(monkeypatch)
    second = _prepare(tmp_path)

    assert calls["n"] == 0, "a matching second call must be a pure cache hit"
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_prepare_chain_crop_frames_cache.py -v`
Expected: the first test FAILS on `assert calls["n"] == 0` (today's `prepare_chain_crop_frames` always rebuilds, so the second call makes 5 real `cv2.imwrite` calls, not 0). The other two tests currently PASS already (rebuilding is today's only behavior); that is fine, they exist to prove Step 3 does not break invalidation.

- [ ] **Step 3: Add the sidecar cache to `pipeline/crop.py`**

Insert these two helpers directly above `prepare_chain_crop_frames` (after line 390's `return out`, before line 393's `def prepare_chain_crop_frames`):

```python
_PREP_META_NAME = "_prep_meta.json"


def _load_matching_prep(view_dir: Path, *, z_range: tuple[int, int], window: dict,
                        anchor_catmaid_z: int
                        ) -> Optional[tuple[str, dict[int, int], int, int]]:
    """None unless ``view_dir`` already holds a complete, matching prepared view.

    Compares the recorded z_range/window/anchor_catmaid_z against what is being
    asked for now, and checks every frame file the sidecar claims is written is
    actually on disk, so a half-built directory (crashed mid-prepare, no sidecar
    or a stale one) is never mistaken for a cache hit."""
    meta_path = view_dir / _PREP_META_NAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    if (list(meta.get("z_range", ())) != list(z_range)
            or meta.get("window") != window
            or int(meta.get("anchor_catmaid_z", -1)) != int(anchor_catmaid_z)):
        return None
    n_frames = int(meta["n_frames"])
    if not all((view_dir / f"{i:05d}.jpg").exists() for i in range(n_frames)):
        return None
    frame_to_z = {int(k): int(v) for k, v in meta["frame_to_z"].items()}
    return str(view_dir), frame_to_z, int(meta["anchor_frame_idx"]), n_frames


def _write_prep_meta(view_dir: Path, *, z_range: tuple[int, int], window: dict,
                     anchor_catmaid_z: int, n_frames: int, anchor_frame_idx: int,
                     frame_to_z: dict[int, int]) -> None:
    """Record what was just written to ``view_dir``, last, via a temp-file
    rename, so a reader never observes a sidecar whose frames are not all
    actually on disk yet (a crash mid-write leaves no sidecar at all)."""
    meta = {"z_range": list(z_range), "window": window,
            "anchor_catmaid_z": int(anchor_catmaid_z), "n_frames": int(n_frames),
            "anchor_frame_idx": int(anchor_frame_idx),
            "frame_to_z": {str(k): int(v) for k, v in frame_to_z.items()}}
    tmp_path = view_dir / (_PREP_META_NAME + ".tmp")
    tmp_path.write_text(json.dumps(meta))
    tmp_path.replace(view_dir / _PREP_META_NAME)
```

Now replace the body of `prepare_chain_crop_frames` (lines 393-462). Old:

```python
def prepare_chain_crop_frames(chain: dict, annotate_df: pd.DataFrame,
                              cw: "alignment.CropWindow", *,
                              frames_root: Optional[Path],
                              anchor_catmaid_z: int,
                              neuron: str, chain_idx: int,
                              frame_store: Optional[FrameStore] = None,
                              z_range: Optional[tuple[int, int]] = None,
                              ) -> tuple[str, dict[int, int], int, int]:
    """Tier-2 video frames: the chain's frames cropped to `cw` and saved as a
    0-indexed JPEG view in `_pcrop` space.

    Each frame is the full-res tif cropped to ``cw.slice_tif()`` then downscaled by
    ``cw.crop_scale``, the SAME crop-then-downscale as ``anchor_crop_predict``, so
    the anchor seed (computed in the crop) and the propagated frames share EXACT
    `_pcrop` pixels. Unlike ``prepare_video_frames`` there is no cross-chain decode
    cache (every chain's window is unique), which makes ``z_range`` (see
    ``prepare_video_frames``'s docstring, same override, e.g.
    ``(anchor_catmaid_z, anchor_catmaid_z)`` for just the anchor) even more worth
    using here: there is no shared cache to fall back on, every z this prepares gets
    decoded fresh, every time. The per-frame read goes through
    ``_read_tif_window``: a windowed memmap slice that pages in only the
    window's rows instead of decoding the whole ~85 MB frame, which is where this
    function's wall-time lived. View dir is namespaced by neuron+chain+crop_scale and
    rebuilt fresh. Returns (view_dir str, frame_to_z, anchor_frame_idx, n_frames).
    """
    import cv2
    import shutil
    from tqdm import tqdm

    if frames_root is None:
        raise ValueError("PipelineConfig.frames_root must be set for video frame prep")

    if z_range is not None:
        start_z, end_z = z_range
    else:
        chain_z = [
            int(annotate_df.loc[
                annotate_df["node_id"].astype(str) == str(n), "z"
            ].item())
            for n in chain["nodes"]
        ]
        start_z, end_z = min(chain_z), max(chain_z)

    fs = frame_store or TifFrameStore()
    anchor_key = fs.key_of_z(anchor_catmaid_z)
    subset = fs.files_in_z_range(start_z, end_z)     # [(key, src_path), ...] sorted by key

    frames_root = Path(frames_root)
    view_dir = (frames_root / "chain_views"
                / f"{neuron}_chain{chain_idx:02d}_pcrop_s{cw.crop_scale}")
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)

    sl = cw.slice_tif()
    frame_to_z: dict[int, int] = {}
    anchor_frame_idx: Optional[int] = None
    for i, (key, src_path) in enumerate(tqdm(subset, desc="caching _pcrop frames", unit="frame")):
        crop = _read_tif_window(src_path, sl)       # windowed read; == cv2.imread(src)[sl]
        crop = _downscale_image(crop, cw.crop_scale)
        cv2.imwrite(str(view_dir / f"{i:05d}.jpg"), crop)
        frame_to_z[i] = fs.z_of_key(key)
        if key == anchor_key:
            anchor_frame_idx = i

    if anchor_frame_idx is None:
        raise AssertionError(
            f"anchor key={anchor_key} not in z-range [{start_z}, {end_z}]"
        )
    return str(view_dir), frame_to_z, anchor_frame_idx, len(subset)
```

New:

```python
def prepare_chain_crop_frames(chain: dict, annotate_df: pd.DataFrame,
                              cw: "alignment.CropWindow", *,
                              frames_root: Optional[Path],
                              anchor_catmaid_z: int,
                              neuron: str, chain_idx: int,
                              frame_store: Optional[FrameStore] = None,
                              z_range: Optional[tuple[int, int]] = None,
                              ) -> tuple[str, dict[int, int], int, int]:
    """Tier-2 video frames: the chain's frames cropped to `cw` and saved as a
    0-indexed JPEG view in `_pcrop` space.

    Each frame is the full-res tif cropped to ``cw.slice_tif()`` then downscaled by
    ``cw.crop_scale``, the SAME crop-then-downscale as ``anchor_crop_predict``, so
    the anchor seed (computed in the crop) and the propagated frames share EXACT
    `_pcrop` pixels. Unlike ``prepare_video_frames`` there is no cross-chain decode
    cache (every chain's window is unique), which makes ``z_range`` (see
    ``prepare_video_frames``'s docstring, same override, e.g.
    ``(anchor_catmaid_z, anchor_catmaid_z)`` for just the anchor) even more worth
    using here. The per-frame read goes through ``_read_tif_window``: a windowed
    memmap slice that pages in only the window's rows instead of decoding the
    whole ~85 MB frame, which is where this function's wall-time lived.

    View dir is namespaced by neuron+chain+crop_scale. A call whose
    (z_range, window, anchor_catmaid_z) exactly matches a sidecar already
    recorded there (``_prep_meta.json``) is a pure cache hit: nothing is read or
    written, the recorded result is returned as-is. This is what lets the
    review GUI's background prefetch (see gui.py's ``_prefetch_next``) do a
    chain's crop ahead of time and have the real, synchronous open reuse it
    instead of redoing the work. Any other call rebuilds the view from scratch,
    exactly as before. Returns (view_dir str, frame_to_z, anchor_frame_idx, n_frames).
    """
    import cv2
    import shutil
    from tqdm import tqdm

    if frames_root is None:
        raise ValueError("PipelineConfig.frames_root must be set for video frame prep")

    if z_range is not None:
        start_z, end_z = z_range
    else:
        chain_z = [
            int(annotate_df.loc[
                annotate_df["node_id"].astype(str) == str(n), "z"
            ].item())
            for n in chain["nodes"]
        ]
        start_z, end_z = min(chain_z), max(chain_z)

    fs = frame_store or TifFrameStore()
    anchor_key = fs.key_of_z(anchor_catmaid_z)
    subset = fs.files_in_z_range(start_z, end_z)     # [(key, src_path), ...] sorted by key

    frames_root = Path(frames_root)
    view_dir = (frames_root / "chain_views"
                / f"{neuron}_chain{chain_idx:02d}_pcrop_s{cw.crop_scale}")

    window = cw.to_dict()
    cached = _load_matching_prep(view_dir, z_range=(start_z, end_z), window=window,
                                 anchor_catmaid_z=anchor_catmaid_z)
    if cached is not None:
        return cached

    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)

    sl = cw.slice_tif()
    frame_to_z: dict[int, int] = {}
    anchor_frame_idx: Optional[int] = None
    for i, (key, src_path) in enumerate(tqdm(subset, desc="caching _pcrop frames", unit="frame")):
        crop = _read_tif_window(src_path, sl)       # windowed read; == cv2.imread(src)[sl]
        crop = _downscale_image(crop, cw.crop_scale)
        cv2.imwrite(str(view_dir / f"{i:05d}.jpg"), crop)
        frame_to_z[i] = fs.z_of_key(key)
        if key == anchor_key:
            anchor_frame_idx = i

    if anchor_frame_idx is None:
        raise AssertionError(
            f"anchor key={anchor_key} not in z-range [{start_z}, {end_z}]"
        )
    _write_prep_meta(view_dir, z_range=(start_z, end_z), window=window,
                     anchor_catmaid_z=anchor_catmaid_z, n_frames=len(subset),
                     anchor_frame_idx=anchor_frame_idx, frame_to_z=frame_to_z)
    return str(view_dir), frame_to_z, anchor_frame_idx, len(subset)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_prepare_chain_crop_frames_cache.py -v`
Expected: PASS, all three tests.

- [ ] **Step 5: Run the full pure-logic test suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same pass count as before this change, plus the 3 new tests). If `tests/test_export_bundle.py`'s mocked-pipeline tests fail, re-check that `_load_matching_prep`/`_write_prep_meta` are not exported anywhere that mock replaces (they are module-private helpers inside `pipeline/crop.py`, not re-exported from `pipeline/__init__.py`, so the existing fake-pipeline monkeypatches in that file are unaffected).

- [ ] **Step 6: Update the frames-tree doc**

In `docs/reference/state-and-storage.md`, replace the "The frames tree" diagram (lines 46-53):

Old:
```
frames_root/
  frames_cache_s<scale>/
    z<file_z>.jpg             # shared decode cache: each EM frame downscaled once, ever
  chain_views/
    <neuron>_chain<idx>_s<scale>/
      00000.jpg ...           # 0-indexed links into the cache, per chain
```

New:
```
frames_root/
  frames_cache_s<scale>/
    z<file_z>.jpg             # shared decode cache: each EM frame downscaled once, ever
  chain_views/
    <neuron>_chain<idx>_s<scale>/
      00000.jpg ...           # 0-indexed links into the cache, per chain
    <neuron>_chain<idx>_pcrop_s<crop_scale>/
      00000.jpg ...           # tier-2: the chain's own crop, decoded fresh (no shared cache)
      _prep_meta.json         # what was prepared (z_range, window, frame_to_z); lets an
                               # identical later call skip the rebuild instead of redoing it
```

Then add one sentence after the existing paragraph below the diagram (after "share a volume."):

```
A tier-2 (`_pcrop`) view has no shared cache behind it, since every chain's crop window is
unique, so `_prep_meta.json` is its own cache: a later call asking for the exact same
z-range, window, and anchor is a no-op instead of a rebuild.
```

- [ ] **Step 7: Run the humanizer skill on the doc change**

Invoke the `humanizer` skill on the new paragraph added to `docs/reference/state-and-storage.md` in Step 6, and apply any fix it suggests before committing.

- [ ] **Step 8: Lint the touched files**

Run: `ruff check pipeline/crop.py tests/test_prepare_chain_crop_frames_cache.py`
Expected: clean (no new warnings). Fix any that appear before committing.

- [ ] **Step 9: Commit**

```bash
git add pipeline/crop.py tests/test_prepare_chain_crop_frames_cache.py docs/reference/state-and-storage.md
git commit -m "pipeline: cache prepare_chain_crop_frames so a repeat call is a no-op

A tier-2 chain's crop view was rebuilt from scratch on every call, even when
nothing had changed since the last one. Adds a _prep_meta.json sidecar
recording what was actually written (z_range, crop window, anchor), checked
before any rebuild. This is the half of the next-chain-prefetch design
(docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md) that
makes a background prefetch's work actually reusable instead of thrown away
by the real open."
```

---

### Task 2: Extract `_load_chain_state_and_cw` and `_peek_chain` in `gui.py`

Pure refactor: no new behavior, only a mechanical extraction that Task 3 builds on. `open_chain` and `_step_chain` must behave identically before and after this task.

**Files:**
- Modify: `gui.py:436-559` (insert `_load_chain_state_and_cw` between `_ensure_local_frames` and `_resolve_chain_frames`)
- Modify: `gui.py:694-734` (`open_chain`'s chain/state/crop-window lookup)
- Modify: `gui.py:1543-1574` (`_step_chain`, gains `_peek_chain`)
- Create: `tests/test_gui_load_chain_state_and_cw.py`
- Create: `tests/test_gui_peek_chain.py`

**Interfaces:**
- Consumes: `pipeline.load_state(path) -> pipeline.ChainState` (existing), `alignment.CropWindow.from_dict(dict) -> CropWindow` (existing), `ReviewContext.find_chain(neuron, chain_idx) -> Optional[dict]` (existing).
- Produces: module-level `_load_chain_state_and_cw(ctx, neuron, chain_idx, chain_dir) -> tuple[Optional[dict], Optional[pipeline.ChainState], Optional[alignment.CropWindow]]` in `gui.py`, used by `open_chain` and (Task 3) `ReviewGUI._prefetch_next`. `ReviewGUI._peek_chain(self, direction: int) -> Optional[tuple[str, int]]`, used by `_step_chain` and (Task 3) `_prefetch_next`.

- [ ] **Step 1: Write the failing tests for `_load_chain_state_and_cw`**

Create `tests/test_gui_load_chain_state_and_cw.py`:

```python
"""_load_chain_state_and_cw: the (chain, state, crop_window) lookup open_chain
and the background prefetch (Task 3) both need, factored out so the two call
sites can't drift. Mirrors the exact short-circuit order open_chain used to
inline: a chain this session's chain list does not have returns (None, None,
None) WITHOUT ever touching state.json, so a scope mismatch stays a clean
"not in this session" signal even if state.json on disk is malformed.

    py -3 -m pytest tests/test_gui_load_chain_state_and_cw.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


def _ctx(*, find_chain):
    return types.SimpleNamespace(find_chain=find_chain)


def test_chain_not_in_session_returns_all_none_without_reading_state(tmp_path):
    chain_dir = tmp_path / "AIZL" / "chain_03"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text("not valid json {{{")   # would raise if read
    ctx = _ctx(find_chain=lambda neuron, idx: None)

    assert gui._load_chain_state_and_cw(ctx, "AIZL", 3, chain_dir) == (None, None, None)


def test_tier2_chain_returns_chain_state_and_crop_window(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_00"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIAL", "chain_idx": 0, "status": "done",
        "anchor_catmaid_z": 1500, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "frames"), "frame_to_z": {"0": 1500}, "n_frames": 1,
        "crop_window": {"origin_tif": [100.0, 200.0], "size_tif": [400, 300],
                        "crop_scale": 2, "sam_scale": 8},
    }))
    stub_chain = {"nodes": ["a", "b"]}
    ctx = _ctx(find_chain=lambda neuron, idx: stub_chain)

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 0, chain_dir)

    assert chain is stub_chain
    assert state.anchor_catmaid_z == 1500
    assert cw.origin_tif == (100.0, 200.0) and cw.crop_scale == 2


def test_legacy_sam_chain_has_no_crop_window(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_01"
    chain_dir.mkdir(parents=True)
    (chain_dir / "state.json").write_text(json.dumps({
        "neuron": "AIAL", "chain_idx": 1, "status": "done",
        "anchor_catmaid_z": 1500, "anchor_frame_idx": 0,
        "frames_dir": str(tmp_path / "frames"), "frame_to_z": {"0": 1500}, "n_frames": 1,
        "crop_window": None,
    }))
    ctx = _ctx(find_chain=lambda neuron, idx: {"nodes": []})

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 1, chain_dir)

    assert chain is not None and state is not None
    assert cw is None


def test_chain_in_session_but_no_state_json_yet(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_02"
    chain_dir.mkdir(parents=True)   # chain dir exists, never run: no state.json
    ctx = _ctx(find_chain=lambda neuron, idx: {"nodes": []})

    chain, state, cw = gui._load_chain_state_and_cw(ctx, "AIAL", 2, chain_dir)

    assert chain is not None
    assert state is None and cw is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_gui_load_chain_state_and_cw.py -v`
Expected: FAIL with `AttributeError: module 'gui' has no attribute '_load_chain_state_and_cw'`.

- [ ] **Step 3: Add `_load_chain_state_and_cw` to `gui.py`**

Insert this module-level function between `_ensure_local_frames` (ends `gui.py:514`) and `_resolve_chain_frames` (starts `gui.py:517`), i.e. in the two blank lines at 515-516:

```python
def _load_chain_state_and_cw(ctx: "ReviewContext", neuron: str, chain_idx: int,
                             chain_dir: Path
                             ) -> tuple[Optional[dict], Optional["pipeline.ChainState"],
                                        Optional["alignment.CropWindow"]]:
    """(chain dict, parsed state.json, tier-2 CropWindow or None) for (neuron,
    chain_idx): the lookup open_chain and the background prefetch (_prefetch_next)
    both need before they can resolve frames.

    ``chain`` is None exactly when this session's chain list does not have
    (neuron, chain_idx) (see open_chain's LookupError for why that matters:
    a scope mismatch, not a missing file). That check happens FIRST and short-
    circuits before state.json is ever read, so a malformed state.json next to
    a chain outside this session's scope still surfaces as "not in this
    session," not as a JSON error."""
    chain = ctx.find_chain(neuron, chain_idx)
    if chain is None:
        return None, None, None
    sp = chain_dir / "state.json"
    state = pipeline.load_state(sp) if sp.exists() else None
    cw = None
    if state is not None and getattr(state, "crop_window", None):
        cw = alignment.CropWindow.from_dict(state.crop_window)
    return chain, state, cw
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_gui_load_chain_state_and_cw.py -v`
Expected: PASS, all four tests.

- [ ] **Step 5: Use the new helper in `open_chain`**

In `gui.py`, replace (lines 702-734):

Old:
```python
        self.chain = self.ctx.find_chain(neuron, chain_idx)
        if self.chain is None:
            # find_chain returning None is a real, tested state (a neuron this
            # session's chains.json/scope does not have), not a bug in it: see
            # test_gui_neuron_filter.py. But letting a None chain flow into frame
            # regeneration crashes three calls later with a bare "'NoneType' object
            # is not subscriptable" deep in prepare_chain_crop_frames, useless for
            # telling a scope mismatch apart from any other failure. The chain DIR
            # existing (checked above) while find_chain fails means chains.json
            # itself is missing this chain or this session was scoped without it.
            raise LookupError(
                f"{neuron} chain_{chain_idx:02d} has a directory at {chain_dir} but "
                f"is not in this session's chain list. Likely causes: this "
                f"session's data/chains.json does not include {neuron}, or the "
                f"launcher scoped this session to a neuron subset that excludes it.")

        # the chain's serialized state carries the ORIGINAL seed (prompts.points_sam
        # / labels / box_sam), loaded so we can pre-populate the prompts layer with
        # it rather than starting empty (else "re-run image phase" has no positive
        # point). Also reused by _anchor_dict. Loaded BEFORE review.load_chain (not
        # after, as before) because frames_dir may need regenerating first, see below.
        sp = chain_dir / "state.json"
        self._state = pipeline.load_state(sp) if sp.exists() else None
        # tier-2 chains were propagated/saved in a per-chain crop space (_pcrop). The
        # CropWindow (persisted in state.json) is what maps _tif skeleton nodes + drives
        # crop-aware QC. The displayed EM/mask/prompts are ALL _pcrop already (frames_dir
        # points at the crop view, masks are crop-sized), so a click is a _pcrop coord and
        # re-predict/resume need no transform: only skeleton/QC/hires consult the window.
        self._cw = None
        if self._state is not None and getattr(self._state, "crop_window", None):
            self._cw = alignment.CropWindow.from_dict(self._state.crop_window)
            print(f"[gui] tier-2 crop chain: _pcrop window {self._cw.size_tif} "
                  f"@ crop_scale {self._cw.crop_scale}")
```

New:
```python
        # the chain's serialized state carries the ORIGINAL seed (prompts.points_sam
        # / labels / box_sam), loaded so we can pre-populate the prompts layer with
        # it rather than starting empty (else "re-run image phase" has no positive
        # point). Also reused by _anchor_dict. Loaded BEFORE review.load_chain (not
        # after, as before) because frames_dir may need regenerating first, see below.
        # tier-2 chains were propagated/saved in a per-chain crop space (_pcrop). The
        # CropWindow (persisted in state.json) is what maps _tif skeleton nodes + drives
        # crop-aware QC. The displayed EM/mask/prompts are ALL _pcrop already (frames_dir
        # points at the crop view, masks are crop-sized), so a click is a _pcrop coord and
        # re-predict/resume need no transform: only skeleton/QC/hires consult the window.
        self.chain, self._state, self._cw = _load_chain_state_and_cw(
            self.ctx, neuron, chain_idx, chain_dir)
        if self.chain is None:
            # find_chain returning None is a real, tested state (a neuron this
            # session's chains.json/scope does not have), not a bug in it: see
            # test_gui_neuron_filter.py. But letting a None chain flow into frame
            # regeneration crashes three calls later with a bare "'NoneType' object
            # is not subscriptable" deep in prepare_chain_crop_frames, useless for
            # telling a scope mismatch apart from any other failure. The chain DIR
            # existing (checked above) while find_chain fails means chains.json
            # itself is missing this chain or this session was scoped without it.
            raise LookupError(
                f"{neuron} chain_{chain_idx:02d} has a directory at {chain_dir} but "
                f"is not in this session's chain list. Likely causes: this "
                f"session's data/chains.json does not include {neuron}, or the "
                f"launcher scoped this session to a neuron subset that excludes it.")
        if self._cw is not None:
            print(f"[gui] tier-2 crop chain: _pcrop window {self._cw.size_tif} "
                  f"@ crop_scale {self._cw.crop_scale}")
```

- [ ] **Step 6: Run the existing open_chain regression test**

Run: `py -3 -m pytest tests/test_gui_open_chain_missing_from_session.py -v`
Expected: PASS, unchanged (this test's fake `ctx.find_chain` returns `None`, so `_load_chain_state_and_cw` now returns `(None, None, None)` immediately, and `open_chain` still raises the same `LookupError` naming `AIZL`).

- [ ] **Step 7: Write the failing tests for `_peek_chain`**

Create `tests/test_gui_peek_chain.py`:

```python
"""_peek_chain: the pure "what would next/prev CHAIN open" lookup, factored
out of _step_chain so the background prefetch (Task 3) can ask the same
question without opening anything or touching GUI state.

    py -3 -m pytest tests/test_gui_peek_chain.py
"""
from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


def _fake_gui(chains, *, neuron, chain_idx):
    return types.SimpleNamespace(
        queue=types.SimpleNamespace(refresh=lambda: None),
        _mode_chains=lambda: list(chains),
        neuron=neuron, chain_idx=chain_idx)


def test_peek_forward_returns_the_next_chain_in_the_list():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 1)


def test_peek_forward_wraps_around_at_the_end():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAR", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 0)


def test_peek_backward_wraps_around_at_the_start():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, -1) == ("AIAR", 0)


def test_peek_returns_none_for_an_empty_queue():
    g = _fake_gui([], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) is None


def test_peek_returns_none_when_the_open_chain_is_the_only_one():
    g = _fake_gui([("AIAL", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) is None


def test_peek_when_the_open_chain_left_the_list_steps_from_the_edge():
    # the current chain was disposed mid-review and no longer appears; forward
    # lands on the first chain of whatever is left, backward on the last
    g = _fake_gui([("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 1)
    assert gui.ReviewGUI._peek_chain(g, -1) == ("AIAR", 0)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 8: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_gui_peek_chain.py -v`
Expected: FAIL with `AttributeError: 'ReviewGUI' object has no attribute '_peek_chain'` (or similar, since `_peek_chain` does not exist yet).

- [ ] **Step 9: Extract `_peek_chain` from `_step_chain`**

In `gui.py`, replace (lines 1549-1574):

Old:
```python
    def _step_chain(self, direction: int) -> None:
        """Cycle to the next/prev CHAIN that still needs a human (different chain, vs
        next/prev *flagged FRAME*, which moves between frames within the open chain).

        Cycles the SAME list the picker shows, per the mode toggle: the pending
        review queue in 'flagged only' mode, or every on-disk chain in 'everything'
        mode. In flagged mode this keeps ``in_review`` chains visible (the fix for
        "can't return to an unfinished chain"): opening a chain marks it in_review, so
        excluding those made the queue look empty as soon as you'd visited each once.
        Only terminal dispositions (approved / rejected / corrected) drop a chain out
        of the flagged list. Wraps around, relative to the chain currently open."""
        self.queue.refresh()
        chains = self._mode_chains()                        # mode-aware (see _mode_chains)
        if not chains:
            print("[gui] no chains to cycle "
                  f"({self._mode_combo.value}); switch mode or refresh")
            return
        cur = (self.neuron, self.chain_idx)
        if cur in chains:
            i = (chains.index(cur) + direction) % len(chains)
        else:
            i = 0 if direction > 0 else len(chains) - 1
        if chains[i] == cur and len(chains) == 1:
            print(f"[gui] {cur[0]} chain {cur[1]:02d} is the only chain in this list")
            return
        self.open_chain(*chains[i])
```

New:
```python
    def _peek_chain(self, direction: int) -> Optional[tuple[str, int]]:
        """The (neuron, chain_idx) `_step_chain(direction)` would open right now,
        without opening it or touching any GUI state. None when there is nothing
        to step to: an empty queue, or the only chain in the list is the one
        already open. Refreshes the queue first, same as _step_chain always did,
        so it reflects the current review state (claims/dispositions since the
        last refresh). Used by _step_chain itself and by the background
        prefetch (_prefetch_next), so both ask the identical question."""
        self.queue.refresh()
        chains = self._mode_chains()                        # mode-aware (see _mode_chains)
        if not chains:
            return None
        cur = (self.neuron, self.chain_idx)
        if cur in chains:
            i = (chains.index(cur) + direction) % len(chains)
        else:
            i = 0 if direction > 0 else len(chains) - 1
        if chains[i] == cur and len(chains) == 1:
            return None
        return chains[i]

    def _step_chain(self, direction: int) -> None:
        """Cycle to the next/prev CHAIN that still needs a human (different chain, vs
        next/prev *flagged FRAME*, which moves between frames within the open chain).

        Cycles the SAME list the picker shows, per the mode toggle: the pending
        review queue in 'flagged only' mode, or every on-disk chain in 'everything'
        mode. In flagged mode this keeps ``in_review`` chains visible (the fix for
        "can't return to an unfinished chain"): opening a chain marks it in_review, so
        excluding those made the queue look empty as soon as you'd visited each once.
        Only terminal dispositions (approved / rejected / corrected) drop a chain out
        of the flagged list. Wraps around, relative to the chain currently open."""
        target = self._peek_chain(direction)
        if target is not None:
            self.open_chain(*target)
            return
        chains = self._mode_chains()        # queue was already refreshed by _peek_chain
        if not chains:
            print("[gui] no chains to cycle "
                  f"({self._mode_combo.value}); switch mode or refresh")
        else:
            cur = (self.neuron, self.chain_idx)
            print(f"[gui] {cur[0]} chain {cur[1]:02d} is the only chain in this list")
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_gui_peek_chain.py -v`
Expected: PASS, all six tests.

- [ ] **Step 11: Run the full pure-logic test suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same pass count as the end of Task 1, plus the new `_load_chain_state_and_cw` and `_peek_chain` tests).

- [ ] **Step 12: Lint the touched files**

Run: `ruff check gui.py tests/test_gui_load_chain_state_and_cw.py tests/test_gui_peek_chain.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 13: Commit**

```bash
git add gui.py tests/test_gui_load_chain_state_and_cw.py tests/test_gui_peek_chain.py
git commit -m "gui: extract _load_chain_state_and_cw and _peek_chain

Pure refactor, no behavior change: open_chain's chain/state/crop-window
lookup and _step_chain's next/prev index math are now standalone, testable
functions. Prep for the background next-chain prefetch
(docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md),
which needs to ask both questions without opening a chain."
```

---

### Task 3: Background prefetch of the next chain

**Files:**
- Modify: `gui.py` (imports, `ReviewGUI.__init__`, `open_chain`'s frame-resolution call and tail, new `_prefetch_next` method)
- Modify: `docs/CHANGELOG.md` (new entry)
- Create: `tests/test_gui_prefetch_next.py`

**Interfaces:**
- Consumes: `ReviewGUI._peek_chain` and `_load_chain_state_and_cw` (Task 2), `_resolve_chain_frames` (existing, unchanged signature).
- Produces: `ReviewGUI._prefetch_next(self) -> None`, called once at the end of `open_chain`. `ReviewGUI._prep_locks: dict[tuple[str, int], threading.Lock]`, created empty in `__init__`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_prefetch_next.py`:

```python
"""_prefetch_next: fires a background prep of whatever `next CHAIN` would open
right now, using a stand-in for threading.Thread that runs synchronously so
the tests are deterministic. See
docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md.

    py -3 -m pytest tests/test_gui_prefetch_next.py
"""
from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


class _SyncThread:
    """Runs its target immediately on .start(), on the calling thread, so
    prefetch tests don't race a real background thread."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _fake_self(tmp_path, *, peek, prep_locks=None):
    return types.SimpleNamespace(
        _peek_chain=lambda direction: peek,
        ctx=types.SimpleNamespace(output_root=tmp_path, cfg="CFG", annotate_df="DF"),
        anchor_only=True, context_frames=2,
        _prep_locks=prep_locks if prep_locks is not None else {})


def test_prefetch_resolves_frames_for_the_peeked_chain(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, "STATE", None))
    monkeypatch.setattr(gui, "_resolve_chain_frames",
                        lambda state, **kw: calls.append((state, kw)) or (None, None, None))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)

    assert len(calls) == 1
    state, kw = calls[0]
    assert state == "STATE"
    assert kw["neuron"] == "AIAL" and kw["chain_idx"] == 1
    assert kw["anchor_only"] is True and kw["context_frames"] == 2
    assert ("AIAL", 1) in fake_self._prep_locks


def test_prefetch_is_a_noop_when_there_is_nothing_to_peek(monkeypatch, tmp_path):
    def _must_not_be_called(*a, **k):
        raise AssertionError("no thread should be started when _peek_chain returns None")
    monkeypatch.setattr(gui.threading, "Thread", _must_not_be_called)

    fake_self = _fake_self(tmp_path, peek=None)
    gui.ReviewGUI._prefetch_next(fake_self)   # must not raise, must not touch threading.Thread


def test_prefetch_is_a_noop_when_the_peeked_chain_has_no_state_yet(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, None, None))
    monkeypatch.setattr(gui, "_resolve_chain_frames",
                        lambda state, **kw: calls.append((state, kw)))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 9))
    gui.ReviewGUI._prefetch_next(fake_self)

    assert calls == [], "a never-run chain (no state.json) has nothing to prefetch"


def test_prefetch_swallows_errors_from_the_background_thread(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)

    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gui, "_load_chain_state_and_cw", _boom)

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)   # must not raise

    out = capsys.readouterr().out
    assert "background prefetch" in out and "AIAL" in out


def test_prefetch_reuses_the_same_lock_object_for_the_same_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, "STATE", None))
    monkeypatch.setattr(gui, "_resolve_chain_frames", lambda state, **kw: (None, None, None))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)
    lock_first = fake_self._prep_locks[("AIAL", 1)]
    gui.ReviewGUI._prefetch_next(fake_self)
    lock_second = fake_self._prep_locks[("AIAL", 1)]

    assert lock_first is lock_second


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_gui_prefetch_next.py -v`
Expected: FAIL with `AttributeError: 'ReviewGUI' object has no attribute '_prefetch_next'`.

- [ ] **Step 3: Add the `threading` import**

In `gui.py`, replace (lines 74-76):

Old:
```python
import argparse
import json
import warnings
```

New:
```python
import argparse
import json
import threading
import warnings
```

- [ ] **Step 4: Add `self._prep_locks` to `ReviewGUI.__init__`**

In `gui.py`, replace (line 688, inside `__init__`):

Old:
```python
        self._recrop_full_hw = None      # (H, W) _tif of the frame shown in the picker
```

New:
```python
        self._recrop_full_hw = None      # (H, W) _tif of the frame shown in the picker
        # one Lock per (neuron, chain_idx) ever opened or prefetched this session, so
        # open_chain and a background _prefetch_next thread can never write the same
        # chain's view directory at once (see _prefetch_next). dict.setdefault on a
        # plain (str, int) key is atomic under the GIL, so this needs no lock of its own.
        self._prep_locks: dict[tuple[str, int], threading.Lock] = {}
```

- [ ] **Step 5: Add `_prefetch_next` and wire it into `open_chain`**

In `gui.py`, replace the frame-resolution call (lines 744-748):

Old:
```python
        frames_dir_override, frame_to_z_override, anchor_idx_override = _resolve_chain_frames(
            self._state, chain=self.chain, cw=self._cw, cfg=self.ctx.cfg,
            annotate_df=self.ctx.annotate_df, neuron=neuron, chain_idx=chain_idx,
            anchor_only=self.anchor_only, context_frames=self.context_frames,
            chain_dir=chain_dir)
```

New:
```python
        lock = self._prep_locks.setdefault((neuron, chain_idx), threading.Lock())
        with lock:
            frames_dir_override, frame_to_z_override, anchor_idx_override = _resolve_chain_frames(
                self._state, chain=self.chain, cw=self._cw, cfg=self.ctx.cfg,
                annotate_df=self.ctx.annotate_df, neuron=neuron, chain_idx=chain_idx,
                anchor_only=self.anchor_only, context_frames=self.context_frames,
                chain_dir=chain_dir)
```

Then replace the end of `open_chain` and add the new method (lines 814-817):

Old:
```python
        print(f"[gui] opened {neuron} chain {chain_idx:02d}: {t} frames, "
              f"{len(self.data.triage_frames)} queued, anchor frame {self.data.anchor_idx}")

    # -- layer builders --------------------------------------------------------
    def _new_prompts_layer(self, scale=(1.0, 1.0, 1.0)):
```

New:
```python
        print(f"[gui] opened {neuron} chain {chain_idx:02d}: {t} frames, "
              f"{len(self.data.triage_frames)} queued, anchor frame {self.data.anchor_idx}")
        self._prefetch_next()

    def _prefetch_next(self) -> None:
        """Fire a background preparation of the chain `next CHAIN` would open
        right now, so by the time you press it the frame crop/decode is
        already done. Depth 1: this only ever looks one hop ahead, re-armed
        every time a chain finishes loading (called at the end of
        open_chain). A target that errors, or that you never actually open,
        is harmless: caught and logged here, never raised into the GUI, and
        an unread prepared view is simply left on disk for next time."""
        target = self._peek_chain(+1)
        if target is None:
            return
        neuron, chain_idx = target
        chain_dir = self.ctx.output_root / neuron / f"chain_{chain_idx:02d}"

        def _run() -> None:
            try:
                chain, state, cw = _load_chain_state_and_cw(
                    self.ctx, neuron, chain_idx, chain_dir)
                if chain is None or state is None:
                    return   # never run yet, or outside this session's scope: nothing to do
                lock = self._prep_locks.setdefault((neuron, chain_idx), threading.Lock())
                with lock:
                    _resolve_chain_frames(
                        state, chain=chain, cw=cw, cfg=self.ctx.cfg,
                        annotate_df=self.ctx.annotate_df, neuron=neuron, chain_idx=chain_idx,
                        anchor_only=self.anchor_only, context_frames=self.context_frames,
                        chain_dir=chain_dir)
            except Exception as e:
                print(f"[gui] background prefetch of {neuron} chain_{chain_idx:02d} "
                      f"failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # -- layer builders --------------------------------------------------------
    def _new_prompts_layer(self, scale=(1.0, 1.0, 1.0)):
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_gui_prefetch_next.py -v`
Expected: PASS, all five tests.

- [ ] **Step 7: Run the full pure-logic test suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same pass count as the end of Task 2, plus the 5 new tests).

- [ ] **Step 8: Write and humanize the CHANGELOG entry**

In `docs/CHANGELOG.md`, add a new bullet to the Contents list, directly after line 25 (`## Contents`) and before the current top entry (line 26):

Old:
```
## Contents
- [2026-09-15, a tracking sheet had typos, so a reprop batch shipped 123 chains short](#r-2026-09-15-reprop-protocol)
```

New:
```
## Contents
- [2026-10-01, the review GUI starts cropping your next chain before you ask for it](#r-2026-10-01-next-chain-prefetch)
- [2026-09-15, a tracking sheet had typos, so a reprop batch shipped 123 chains short](#r-2026-09-15-reprop-protocol)
```

Then add the new dated section directly after line 64's `---` separator and before line 66's `<a id="r-2026-09-15-reprop-protocol"></a>`:

```markdown
<a id="r-2026-10-01-next-chain-prefetch"></a>
## 2026-10-01, the review GUI starts cropping your next chain before you ask for it

Opening the next chain in `--anchor-only` review always blocked on re-cropping that chain's
frames, even for a window as small as the anchor plus two frames of context. The real cost
was `prepare_chain_crop_frames`: a tier-2 chain has no cross-chain decode cache, and the
function rebuilt its view directory from scratch on every call, even when an identical one
had just been built.

Two changes. `prepare_chain_crop_frames` now writes a small sidecar recording what it
actually built (z-range, crop window, anchor), and skips the rebuild entirely when a later
call asks for the exact same thing. On top of that, `gui.py`'s `open_chain` now fires a
background thread, once a chain finishes loading, that prepares whatever `next CHAIN` would
open next, using the session's own `--anchor-only`/`--context-frames` settings. A lock per
`(neuron, chain_idx)` keeps that thread and a later real open from writing the same
directory at once. No new flag: this is always on, and a mistargeted or failed prefetch is
silently harmless.

Full design in
[next-chain-frame-prefetch-design.md](superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md).
```

Now invoke the `humanizer` skill on this new CHANGELOG section (both the Contents bullet and the dated entry) and apply any fix it suggests before committing.

- [ ] **Step 9: Lint the touched files**

Run: `ruff check gui.py tests/test_gui_prefetch_next.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 10: Manual smoke check (optional but recommended before committing)**

This step needs a real output tree and is not part of the automated test suite; skip it if none is available in the current environment, but run it before treating the feature as done in practice.

Run: `py -3 gui.py --output-root "<a manual_verify_* tree>" --neuron <NEURON> --anchor-only --context-frames 2`, open a chain, then press the "next CHAIN ⇉" button. Confirm in the console output that a `[gui] opened ...` line for the chain you just pressed into appears with no multi-second pause, and that no `[gui] background prefetch ... failed` line appears during normal use.

- [ ] **Step 11: Commit**

```bash
git add gui.py tests/test_gui_prefetch_next.py docs/CHANGELOG.md
git commit -m "gui: prefetch the next chain's frames in the background

open_chain now fires a daemon thread, once it finishes loading a chain,
that prepares whichever chain next CHAIN would open next, reusing the
session's own anchor_only/context_frames. A per-(neuron, chain_idx) lock
keeps it from colliding with a later real open of the same chain. Depends
on the prepare_chain_crop_frames cache landed earlier so the prefetch's
work is not thrown away when you actually open that chain.

Closes the latency the next-chain-frame-prefetch design set out to fix:
docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md."
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** Part 1 (idempotent cache) -> Task 1. Part 2 (background prefetch) and its concurrency section -> Task 3. The `_peek_chain` DRY requirement the spec calls out explicitly -> Task 2. Non-goals (no CLI flag, depth 1, no backward prefetch) are respected throughout: nothing in any task adds a flag, and `_prefetch_next` only ever calls `_peek_chain(+1)` once.
- **Lock key simplification:** the spec's prose says a lock "per `(neuron, chain_idx, scale_or_crop_scale)`"; this plan locks on `(neuron, chain_idx)` alone, documented in Global Constraints above, since `self.anchor_only`/`self.ctx.cfg` are fixed for a `ReviewGUI` session's lifetime, so the scale component can never actually vary for a given chain within one session.
- **Type consistency checked:** `_load_chain_state_and_cw`'s return order `(chain, state, cw)` matches every call site (Task 2's `open_chain` edit, Task 3's `_prefetch_next`, and all new tests). `_peek_chain`'s `Optional[tuple[str, int]]` return matches how both `_step_chain` and `_prefetch_next` consume it (`if target is None: ...` / `neuron, chain_idx = target`).
