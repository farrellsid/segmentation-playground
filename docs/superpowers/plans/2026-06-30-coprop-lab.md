# Co-propagation lab implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `coprop_lab.py`, a standalone napari app that tests whether co-segmenting a chain's neighbors changes the target mask, comparing a neighbors-off baseline against a neighbors-on treatment, visually and without saving anything.

**Architecture:** One top-level driver file imports the pipeline library for the science (chain loading, prompt building, image-mode prediction) and napari for the viewer. A small `MultiObjectCopropSession` (ported from the reverted `feat/multi-instance-coprop` branch) seeds N objects in one `inference_state` and toggles SAM2's non-overlap flags. The two questions map onto the two flags: output-only (`non_overlap_masks`) for segmentation cleanup, memory (`non_overlap_masks_for_mem_enc`) for propagation.

**Tech Stack:** Python 3, SAM2 (video + image predictors), napari 0.7.0, numpy, OpenCV (frame reads via `sam2_utils.video_viz._load_frame`).

## Global Constraints

- No em dashes anywhere (code, comments, docstrings, commit messages). Use commas, colons, parentheses, or separate sentences.
- `coprop_lab.py` is a driver (like `gui.py` / `batch.py`); it may import `pipeline` and `sam2_utils`. The library must never import it. `tests/test_import_direction.py` enforces the library direction.
- Module-level imports in `coprop_lab.py` must be torch-free and napari-free. Import torch and napari inside the functions that need them, so `import coprop_lab` stays CPU-only for the pure-helper tests (the same tactic `gui.py` uses).
- CPU tests are torch-free and run with `py -3 -m pytest`. Keep new pure-logic tests torch-free.
- Lint with `ruff check .`. Clean only the files you touch.
- Commit incrementally, one concern per commit.
- Run the `humanizer` skill on any prose committed (this plan's own docs are already done; no other prose ships in this feature).
- The pipeline library (`pipeline/`, `sam2_utils/`) is not modified by this work.

---

### Task 1: Pure layer helpers and their tests

The only torch-free, unit-testable logic: turning per-object mask dicts into napari label stacks and a diff stack. Ported from the reverted branch's `gui._neighbor_label_stack` / `_diff_stack` / `_label_stack_from_segments`, generalized to a single `label_stack` over any set of object ids.

**Files:**
- Create: `coprop_lab.py`
- Test: `tests/test_coprop_lab.py`

**Interfaces:**
- Produces:
  - `label_stack(segments: dict[int, dict[int, np.ndarray]], obj_ids: list[int], t: int, hw: tuple[int, int]) -> np.ndarray` returns `(t, H, W)` uint8, each object id painted with its own id, background 0, squeezing the `(1, H, W)` SAM2 slice.
  - `build_diff_stack(alone: np.ndarray, withn: np.ndarray, obj_id: int) -> np.ndarray` returns `(t, H, W)` uint8: 1 where the target had a pixel alone but lost it, 2 where it gained one.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coprop_lab.py`:

```python
"""Unit tests for coprop_lab's pure label/diff helpers.

Torch-free and napari-free: coprop_lab imports torch and napari only lazily, so
importing the module and exercising these helpers needs no GPU and no viewer.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import coprop_lab


def _mask3d(hw, box):
    """A (1, H, W) bool mask (the SAM2 video-logits shape) with [y0,y1,x0,x1] filled."""
    H, W = hw
    m = np.zeros((1, H, W), dtype=bool)
    y0, y1, x0, x1 = box
    m[0, y0:y1, x0:x1] = True
    return m


def test_label_stack_squeezes_and_paints_each_requested_obj():
    H, W, t = 10, 12, 2
    segments = {
        0: {1: _mask3d((H, W), (0, 2, 0, 2)),
            2: _mask3d((H, W), (0, 3, 0, 3)),
            3: _mask3d((H, W), (5, 8, 5, 8))},
        1: {2: _mask3d((H, W), (1, 2, 1, 2))},
    }
    out = coprop_lab.label_stack(segments, [2, 3], t, (H, W))
    assert out.shape == (t, H, W)          # no IndexError on the (1,H,W) masks
    assert out[0, 0, 0] == 2                # neighbor 2 painted with its id
    assert out[0, 6, 6] == 3                # neighbor 3 painted with its id
    assert out[0, 9, 9] == 0                # background stays 0
    assert out[1, 1, 1] == 2
    assert not (out == 1).any()             # obj 1 not requested, so never painted


def test_label_stack_single_target_obj():
    H, W, t = 4, 5, 1
    segments = {0: {1: _mask3d((H, W), (0, 2, 0, 2))}}
    out = coprop_lab.label_stack(segments, [1], t, (H, W))
    assert out[0, 0, 0] == 1
    assert out[0, 3, 3] == 0


def test_build_diff_stack_marks_lost_and_gained():
    H, W, t = 4, 4, 1
    obj = 1
    alone = np.zeros((t, H, W), dtype=np.uint8)
    withn = np.zeros((t, H, W), dtype=np.uint8)
    alone[0, 0, 0] = obj            # present alone, absent with neighbors -> lost
    withn[0, 2, 2] = obj            # absent alone, present with neighbors -> gained
    alone[0, 1, 1] = obj            # present in both -> unchanged
    withn[0, 1, 1] = obj
    diff = coprop_lab.build_diff_stack(alone, withn, obj)
    assert diff[0, 0, 0] == 1        # lost
    assert diff[0, 2, 2] == 2        # gained
    assert diff[0, 1, 1] == 0        # unchanged
    assert diff[0, 3, 3] == 0        # background


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_coprop_lab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coprop_lab'` (the file does not exist yet).

- [ ] **Step 3: Write the minimal implementation**

Create `coprop_lab.py` with the module header and the two helpers (no torch, no napari at module scope):

```python
"""Co-propagation lab: a standalone, disposable napari test for the neighbor-competition
hypothesis. Not part of the pipeline; saves nothing, scores nothing. See
docs/superpowers/specs/2026-06-30-coprop-lab-design.md.

torch and napari are imported lazily inside the functions that need them, so importing
this module for the pure helpers stays CPU-only.
"""

from __future__ import annotations

import numpy as np


def label_stack(segments, obj_ids, t, hw):
    """(t, H, W) uint8: each obj_id in `obj_ids` painted with its own id, background 0.

    `segments` is {frame_idx: {obj_id: mask}} where a mask is bool HxW or the SAM2
    video shape (1, H, W); the leading dim is squeezed. Used for the target-alone, the
    target-with-neighbors, and the neighbors layers, by passing the relevant id list.
    """
    H, W = hw
    out = np.zeros((t, H, W), dtype=np.uint8)
    for fi, seg in segments.items():
        if not (0 <= fi < t):
            continue
        for oid in obj_ids:
            if oid in seg:
                m = np.asarray(seg[oid])
                m = m[0] if m.ndim == 3 else m   # SAM2 logits carry a leading (1,H,W) dim
                if m.shape == (H, W):
                    out[fi][m.astype(bool)] = oid
    return out


def build_diff_stack(alone, withn, obj_id):
    """(t, H, W) uint8: 1 where the target had a pixel ALONE but lost it with neighbors
    (bleed carved out), 2 where it GAINED one. The visual read of the test. Under the
    output-only variant the target can only lose pixels, so a correct Test 1 diff has no 2s.
    """
    a = (alone == obj_id)
    b = (withn == obj_id)
    out = np.zeros_like(alone, dtype=np.uint8)
    out[a & ~b] = 1     # lost (carved-out bleed)
    out[b & ~a] = 2     # gained
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_coprop_lab.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check coprop_lab.py tests/test_coprop_lab.py
git add coprop_lab.py tests/test_coprop_lab.py
git commit -m "coprop-lab: pure label_stack + build_diff_stack helpers with tests"
```

---

### Task 2: Chain loader and EM frame stack

Load an already-run chain into memory for the viewer. This is a thin wrapper over the existing `review.load_chain` and `pipeline.load_state`, plus an eager frame-stack reader for display. No torch.

**Files:**
- Modify: `coprop_lab.py`
- Test: `tests/test_coprop_lab.py` (add the frame-stack reader test)

**Interfaces:**
- Consumes: `sam2_utils.review.load_chain`, `pipeline.load_state`, `sam2_utils.video_viz._load_frame`, `sam2_utils.alignment.CropWindow`.
- Produces:
  - `load_em_stack(frames_dir, n_frames) -> np.ndarray` returns `(n, H, W, 3)` uint8 RGB.
  - `LabChain` dataclass with fields: `em` (ndarray `(T,H,W,3)`), `frames_dir` (str), `anchor_idx` (int), `obj_id` (int), `frame_to_z` (dict[int,int]), `target_prompts` (`pipeline.Prompts`), `anchor_mask` (bool `(H,W)`), `n_frames` (int), `hw` (tuple), `crop_window` (optional).
  - `load_lab_chain(output_root, neuron, chain_idx) -> LabChain`.

- [ ] **Step 1: Write the failing test for the frame reader**

Add to `tests/test_coprop_lab.py`:

```python
def test_load_em_stack_reads_indexed_jpegs(tmp_path):
    import cv2
    # two 4x6 BGR frames named like the chain's 0-indexed jpegs
    for i in range(2):
        img = np.full((4, 6, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), img)
    stack = coprop_lab.load_em_stack(str(tmp_path), 2)
    assert stack.shape == (2, 4, 6, 3)
    assert stack.dtype == np.uint8
```

Note: `sam2_utils.video_viz._load_frame(frames_dir, idx)` reads `{idx:05d}.jpg` (verified), which is why the test writes `{i:05d}.jpg`. If a chain on disk uses a different naming, the Step 4 smoke check will surface it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_coprop_lab.py::test_load_em_stack_reads_indexed_jpegs -v`
Expected: FAIL with `AttributeError: module 'coprop_lab' has no attribute 'load_em_stack'`.

- [ ] **Step 3: Implement the loaders**

Add to `coprop_lab.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_em_stack(frames_dir, n_frames):
    """(n, H, W, 3) uint8 RGB over the chain's 0-indexed frames, eager. Reuses the same
    single-frame reader review/video_viz use, so naming and color match the rest of the repo."""
    from sam2_utils.video_viz import _load_frame
    frames = [_load_frame(Path(frames_dir), i) for i in range(n_frames)]
    return np.stack(frames, axis=0)


@dataclass
class LabChain:
    em: np.ndarray                  # (T, H, W, 3) uint8 RGB, the display stack
    frames_dir: str                 # path SAM2 init_state needs
    anchor_idx: int
    obj_id: int                     # the target object id (== 1 in production)
    frame_to_z: dict
    target_prompts: object          # pipeline.Prompts (box + points in propagation space)
    anchor_mask: np.ndarray         # bool (H, W), the saved target mask at the anchor frame
    n_frames: int
    hw: tuple
    crop_window: Optional[object] = None


def load_lab_chain(output_root, neuron, chain_idx):
    """Build a LabChain from an on-disk, already-run chain. Pure I/O, no torch."""
    import pipeline
    from sam2_utils import review, alignment

    chain_dir = Path(output_root) / neuron / f"chain_{int(chain_idx):02d}"
    if not chain_dir.exists():
        raise FileNotFoundError(f"no chain dir at {chain_dir}")

    data = review.load_chain(chain_dir, verbose=True)        # ReviewData
    t = (max(data.video_segments) + 1) if data.video_segments else 0

    any_mask = next(iter(data.video_segments.values()))[data.obj_id]
    any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
    H, W = np.asarray(any_mask).shape

    anchor_seg = data.video_segments.get(data.anchor_idx, {})
    am = anchor_seg.get(data.obj_id)
    if am is None:
        raise ValueError(f"no target mask at anchor frame {data.anchor_idx}")
    am = np.asarray(am)
    anchor_mask = (am[0] if am.ndim == 3 else am).astype(bool)

    state_path = chain_dir / "state.json"
    state = pipeline.load_state(state_path) if state_path.exists() else None
    prompts = getattr(state, "prompts", None)
    cw = None
    if state is not None and getattr(state, "crop_window", None):
        cw = alignment.CropWindow.from_dict(state.crop_window)

    em = load_em_stack(data.frames_dir, t)
    return LabChain(em=em, frames_dir=str(data.frames_dir), anchor_idx=int(data.anchor_idx),
                    obj_id=int(data.obj_id), frame_to_z=data.frame_to_z,
                    target_prompts=prompts, anchor_mask=anchor_mask,
                    n_frames=t, hw=(H, W), crop_window=cw)
```

- [ ] **Step 4: Run the frame-reader test, then smoke-test the full loader on a real chain**

Run: `py -3 -m pytest tests/test_coprop_lab.py::test_load_em_stack_reads_indexed_jpegs -v`
Expected: PASS (if it fails on the jpeg name, inspect a real chain's `frames_dir` listing and `sam2_utils/video_viz.py:_load_frame` to match the naming, then re-run).

Smoke (needs a real chain on disk; substitute a neuron/chain that exists under the eval or production output root):

```bash
py -3 -c "import coprop_lab as c; lc = c.load_lab_chain(r'data/groundtruth/pred_p280/batch_masks_multichain', 'AVAL', 7); print('em', lc.em.shape, 'anchor', lc.anchor_idx, 'obj', lc.obj_id, 'hw', lc.hw, 'mask px', int(lc.anchor_mask.sum()), 'cw', lc.crop_window is not None)"
```
Expected: prints a 4-D `em` shape, an integer anchor frame, `obj 1`, the `(H, W)`, a non-zero anchor-mask pixel count, and whether it is a tier-2 crop chain. Adjust the output root and neuron/chain to one that exists in your tree.

- [ ] **Step 5: Lint and commit**

```bash
ruff check coprop_lab.py tests/test_coprop_lab.py
git add coprop_lab.py tests/test_coprop_lab.py
git commit -m "coprop-lab: LabChain loader + eager EM frame stack"
```

---

### Task 3: MultiObjectCopropSession

The torch-touching propagation session, ported from the reverted branch and extended with a mask-seed path for the correct-seed (Test 2) case. Seeds N objects in one `inference_state`, toggles the non-overlap flags, restores them on close.

**Files:**
- Modify: `coprop_lab.py`

**Interfaces:**
- Consumes: a SAM2 video predictor, `pipeline.Prompts`.
- Produces: class `MultiObjectCopropSession(video_predictor, frames_dir, *, non_overlap=False, non_overlap_mem_enc=False, offload_video_to_cpu=True)` with:
  - `seed_points_box(obj_id, prompts, anchor_frame_idx, *, seed_box=True, seed_points=True, seed_negatives=False)`
  - `seed_mask(obj_id, mask, anchor_frame_idx)` (mask is bool `(H, W)`)
  - `run_bidirectional()`
  - attribute `video_segments: {frame_idx: {obj_id: bool mask (1,H,W)}}`
  - `close()` and context-manager support.

- [ ] **Step 1: Implement the session**

Add to `coprop_lab.py`:

```python
class MultiObjectCopropSession:
    """Co-propagate several objects (target + neighbors) in ONE inference_state.

    SAM2 tracks each object's memory independently; the only coupling is the per-pixel
    non-overlap argmax. This session seeds N objects and toggles that argmax via the two
    predictor flags (set at runtime, RESTORED on close so a shared predictor is never left
    mutated):
      non_overlap          -> non_overlap_masks (OUTPUT masks only; Test 1 cleanup)
      non_overlap_mem_enc  -> non_overlap_masks_for_mem_enc (fed back into memory, changes
                              the trajectory; Test 2 propagation).
    A no-op with a single object: the constraint needs at least two objects to do anything.
    """

    def __init__(self, video_predictor, frames_dir, *, non_overlap=False,
                 non_overlap_mem_enc=False, offload_video_to_cpu=True):
        self.vp = video_predictor
        self.obj_ids = []
        self._orig_no = getattr(video_predictor, "non_overlap_masks", False)
        self._orig_no_mem = getattr(video_predictor, "non_overlap_masks_for_mem_enc", False)
        video_predictor.non_overlap_masks = bool(non_overlap)
        video_predictor.non_overlap_masks_for_mem_enc = bool(non_overlap_mem_enc)
        self.inference_state = video_predictor.init_state(
            video_path=str(frames_dir), offload_video_to_cpu=offload_video_to_cpu)
        video_predictor.reset_state(self.inference_state)
        self.video_segments = {}
        self._closed = False

    def seed_points_box(self, obj_id, prompts, anchor_frame_idx, *, seed_box=True,
                        seed_points=True, seed_negatives=False):
        """Seed one object from a Prompts (box + positive point by default)."""
        pts = np.asarray(prompts.points_sam, dtype=np.float32)
        labels = np.asarray(prompts.labels, dtype=np.int32)
        keep = np.ones(len(labels), dtype=bool)
        if not seed_points:
            keep &= labels != 1
        if not seed_negatives:
            keep &= labels != 0
        pts, labels = pts[keep], labels[keep]
        box = (np.asarray(prompts.box_sam, dtype=np.float32)
               if (seed_box and prompts.box_sam is not None) else None)
        if box is None and len(pts) == 0:
            raise ValueError("empty seed: enable at least one of box / points")
        # SAM2 requires clear_old_points=True when a box is present (box must precede points).
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state, frame_idx=int(anchor_frame_idx),
            obj_id=int(obj_id), box=box, points=(pts if len(pts) else None),
            labels=(labels if len(labels) else None))
        if int(obj_id) not in self.obj_ids:
            self.obj_ids.append(int(obj_id))

    def seed_mask(self, obj_id, mask, anchor_frame_idx):
        """Seed one object from a 2D bool mask (the correct-seed path, Test 2)."""
        self.vp.add_new_mask(
            inference_state=self.inference_state, frame_idx=int(anchor_frame_idx),
            obj_id=int(obj_id), mask=np.asarray(mask, dtype=bool))
        if int(obj_id) not in self.obj_ids:
            self.obj_ids.append(int(obj_id))

    def _drain(self, *, reverse):
        for f, obj_ids, mask_logits in self.vp.propagate_in_video(
                self.inference_state, reverse=reverse):
            fi = int(f)
            per_obj = self.video_segments.setdefault(fi, {})
            for i, oid in enumerate(obj_ids):
                per_obj[int(oid)] = (mask_logits[i].cpu().numpy() > 0.0)

    def run_bidirectional(self):
        """Forward then reverse over all seeded objects (one shared memory)."""
        self._drain(reverse=False)
        self._drain(reverse=True)

    def close(self):
        if not self._closed:
            self.vp.non_overlap_masks = self._orig_no
            self.vp.non_overlap_masks_for_mem_enc = self._orig_no_mem
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 2: Smoke-test the session on a real chain (needs GPU)**

```bash
py -3 -c "
import numpy as np, coprop_lab as c
from sam2_utils.setup import build_predictor
lc = c.load_lab_chain(r'data/groundtruth/pred_p280/batch_masks_multichain', 'AVAL', 7)
vp, dev = build_predictor(kind='video')
with c.MultiObjectCopropSession(vp, lc.frames_dir) as s:
    s.seed_mask(1, lc.anchor_mask, lc.anchor_idx)
    s.run_bidirectional()
    tgt = sum(1 for seg in s.video_segments.values() if 1 in seg)
    print('frames with target mask:', tgt, '/', lc.n_frames)
"
```
Expected: prints a frame count close to `n_frames` (the target propagated through most of the stack). Single object, so the non-overlap flags are inert here; this just proves seeding and propagation collect correctly.

- [ ] **Step 3: Lint and commit**

```bash
ruff check coprop_lab.py
git add coprop_lab.py
git commit -m "coprop-lab: MultiObjectCopropSession (point/box + mask seeds, flag toggles)"
```

---

### Task 4: Neighbor prediction from a click

Turn a click point in the chain's propagation space into a neighbor seed: run image-mode SAM2 at that point, derive a box, return both for `seed_points_box`. Operates entirely in the display/propagation space (the click coordinate and the frame image are both already in that space), so there is no coordinate transform.

**Files:**
- Modify: `coprop_lab.py`

**Interfaces:**
- Consumes: a SAM2 image predictor, `pipeline.image_predict`, `pipeline.box_from_mask`, `pipeline.Prompts`.
- Produces: `predict_neighbor_at(image_predictor, frame_img, xy, *, box_margin=6) -> (mask: bool (H,W), prompts: pipeline.Prompts) | None`. `frame_img` is the RGB display frame `(H, W, 3)`; `xy` is `(x, y)` in that frame's pixels. Returns `None` if the predicted mask is empty.

- [ ] **Step 1: Implement the helper**

Add to `coprop_lab.py`:

```python
def predict_neighbor_at(image_predictor, frame_img, xy, *, box_margin=6):
    """Image-mode predict a neighbor mask from a single positive click in propagation space.

    Returns (mask bool HxW, Prompts with box + the positive point), or None if SAM2 returns
    an empty mask. The returned Prompts is ready for MultiObjectCopropSession.seed_points_box.
    """
    import pipeline
    x, y = float(xy[0]), float(xy[1])
    point_prompts = pipeline.Prompts(
        points_sam=np.asarray([[x, y]], dtype=float),
        labels=np.asarray([1], dtype=int), box_sam=None)
    mask, _score, _logits = pipeline.image_predict(image_predictor, frame_img, point_prompts)
    if mask is None or not mask.any():
        return None
    box = pipeline.box_from_mask(mask, margin=box_margin, image_hw_sam=mask.shape[:2])
    if box is None:
        return None
    seed = pipeline.Prompts(
        points_sam=np.asarray([[x, y]], dtype=float),
        labels=np.asarray([1], dtype=int),
        box_sam=np.asarray(box, dtype=np.float32))
    return mask.astype(bool), seed
```

- [ ] **Step 2: Smoke-test neighbor prediction on a real chain (needs GPU)**

```bash
py -3 -c "
import coprop_lab as c
from sam2_utils.setup import build_predictor
lc = c.load_lab_chain(r'data/groundtruth/pred_p280/batch_masks_multichain', 'AVAL', 7)
ip, dev = build_predictor(kind='image')
H, W = lc.hw
frame = lc.em[lc.anchor_idx]
res = c.predict_neighbor_at(ip, frame, (W // 2, H // 2))
print('none' if res is None else ('mask px %d, box %s' % (int(res[0].sum()), res[1].box_sam)))
"
```
Expected: prints a non-zero mask pixel count and a 4-number box (a mask near the frame center), or `none` if the center happens to be background. Try a few `xy` values over visible cells if the first lands on background.

- [ ] **Step 3: Lint and commit**

```bash
ruff check coprop_lab.py
git add coprop_lab.py
git commit -m "coprop-lab: predict_neighbor_at (click -> image-mode mask + box)"
```

---

### Task 5: The viewer app and CLI

Wire everything into a napari app: EM stack, a paintable target mask, click-to-seed neighbors with an immediate preview, the seed-source and variant controls plus the two presets, the run-A/B driver, and the result and diff layers with a pixel readout. Guarded to refuse the run until at least one neighbor is seeded.

**Files:**
- Modify: `coprop_lab.py`

**Interfaces:**
- Consumes: everything above, `sam2_utils.setup.build_predictor`, napari, `label_stack`, `build_diff_stack`.
- Produces: class `CopropLab(output_root, neuron, chain_idx)` with `.run()` (opens the viewer), and a `main()` CLI entry. `CopropLab.run_ab()` performs the two passes and refreshes the result layers.

- [ ] **Step 1: Implement the app**

Add to `coprop_lab.py`. The viewer scales every overlay to the EM by the width ratio (mask px to EM px), matching `gui.py`'s convention so a click maps back to mask coordinates:

```python
class CopropLab:
    """Standalone napari app for the co-propagation A/B. Saves nothing.

    Controls (in the dock):
      - target seed:  'auto-prompts' (the saved box+point) or 'current mask' (the paint layer)
      - variant:      'output-only' (Test 1 cleanup) or 'memory' (Test 2 propagation)
      - presets:      Test 1 sets (auto-prompts, output-only); Test 2 sets (current mask, memory)
      - click the EM to seed a neighbor (image-mode preview shown immediately)
      - 'remove last neighbor', 'run A/B'
    Result layers: target (alone), target (w/ neighbors), neighbors, diff (1=lost, 2=gained).
    """

    NEIGHBOR_BASE_ID = 2     # target is obj 1; neighbors are 2, 3, ...

    def __init__(self, output_root, neuron, chain_idx):
        self.lc = load_lab_chain(output_root, neuron, chain_idx)
        self.neuron, self.chain_idx = neuron, int(chain_idx)
        self.target_seed = "auto-prompts"
        self.variant = "output-only"
        self.neighbors = []         # list of dicts: {"obj_id", "prompts", "mask"}
        self.image_predictor = None
        self.video_predictor = None
        self.viewer = None
        self._em_world = 1.0        # EM px per mask px (width ratio)

    # -- predictors (lazy; built once) ----------------------------------------
    def _ensure_predictors(self):
        from sam2_utils.setup import build_predictor
        if self.video_predictor is None:
            self.video_predictor, dev = build_predictor(kind="video")
            self.image_predictor, _ = build_predictor(kind="image", device=dev)

    # -- viewer ---------------------------------------------------------------
    def run(self):
        import napari
        lc = self.lc
        H, W = lc.hw
        self._em_world = float(lc.em.shape[2]) / float(W) if W else 1.0
        s = self._em_world
        lscale = (1.0, s, s)

        self.viewer = napari.Viewer(title=f"coprop lab: {self.neuron} chain {self.chain_idx:02d}")
        self.viewer.add_image(lc.em, name="EM", rgb=True)

        # target paint layer, preloaded with the saved anchor mask at the anchor frame
        paint = np.zeros((lc.n_frames, H, W), dtype=np.uint8)
        paint[lc.anchor_idx][lc.anchor_mask] = lc.obj_id
        self._paint = self.viewer.add_labels(paint, name="target seed (paint)",
                                             opacity=0.5, scale=lscale)
        self._neighbors_layer = self.viewer.add_labels(
            np.zeros((lc.n_frames, H, W), dtype=np.uint8), name="neighbor seeds",
            opacity=0.5, scale=lscale)

        self._em_layer = self.viewer.layers["EM"]
        self._em_layer.mouse_drag_callbacks.append(self._on_click)

        self._build_dock()
        self.viewer.dims.set_current_step(0, int(lc.anchor_idx))
        napari.run()

    def _build_dock(self):
        from magicgui.widgets import ComboBox, PushButton, Label, Container
        seed_cb = ComboBox(label="target seed", choices=["auto-prompts", "current mask"],
                           value=self.target_seed)
        var_cb = ComboBox(label="variant", choices=["output-only", "memory"],
                          value=self.variant)
        seed_cb.changed.connect(lambda v: setattr(self, "target_seed", v))
        var_cb.changed.connect(lambda v: setattr(self, "variant", v))

        t1 = PushButton(text="preset: Test 1 (cleanup)")
        t2 = PushButton(text="preset: Test 2 (propagation)")
        t1.clicked.connect(lambda: self._set_preset(seed_cb, var_cb, "auto-prompts", "output-only"))
        t2.clicked.connect(lambda: self._set_preset(seed_cb, var_cb, "current mask", "memory"))

        rm = PushButton(text="remove last neighbor")
        run = PushButton(text="run A/B")
        rm.clicked.connect(self._remove_last_neighbor)
        run.clicked.connect(lambda: self.run_ab())
        self._status = Label(label="neighbors", value="0 seeded")

        box = Container(widgets=[seed_cb, var_cb, t1, t2, rm, run, self._status])
        self.viewer.window.add_dock_widget(box, name="coprop", area="right")

    def _set_preset(self, seed_cb, var_cb, seed, variant):
        seed_cb.value = seed
        var_cb.value = variant
        self.target_seed, self.variant = seed, variant

    # -- neighbor seeding by click --------------------------------------------
    def _on_click(self, layer, event):
        # only seed on the anchor frame, in add-by-click (left button, no drag)
        if int(self.viewer.dims.current_step[0]) != int(self.lc.anchor_idx):
            print("[coprop] click on the anchor frame to seed a neighbor")
            return
        self._ensure_predictors()
        pos = self._em_layer.world_to_data(event.position)   # (z, y, x) in EM world
        y, x = float(pos[1]) / self._em_world, float(pos[2]) / self._em_world
        res = predict_neighbor_at(self.image_predictor, self.lc.em[self.lc.anchor_idx], (x, y))
        if res is None:
            print(f"[coprop] empty neighbor mask at ({x:.0f},{y:.0f}); try another spot")
            return
        mask, prompts = res
        obj_id = self.NEIGHBOR_BASE_ID + len(self.neighbors)
        self.neighbors.append({"obj_id": obj_id, "prompts": prompts, "mask": mask})
        self._refresh_neighbor_preview()
        print(f"[coprop] neighbor {obj_id} seeded ({int(mask.sum())} px)")

    def _remove_last_neighbor(self):
        if self.neighbors:
            dropped = self.neighbors.pop()
            self._refresh_neighbor_preview()
            print(f"[coprop] removed neighbor {dropped['obj_id']}")

    def _refresh_neighbor_preview(self):
        H, W = self.lc.hw
        vol = np.zeros((self.lc.n_frames, H, W), dtype=np.uint8)
        for nb in self.neighbors:
            vol[self.lc.anchor_idx][nb["mask"]] = nb["obj_id"]
        self._neighbors_layer.data = vol
        self._status.value = f"{len(self.neighbors)} seeded"

    # -- the A/B run ----------------------------------------------------------
    def _seed_all(self, session):
        lc = self.lc
        if self.target_seed == "current mask":
            tgt = (self._paint.data[lc.anchor_idx] == lc.obj_id)
            if not tgt.any():
                raise ValueError("target paint layer is empty at the anchor frame")
            session.seed_mask(lc.obj_id, tgt, lc.anchor_idx)
        else:
            if lc.target_prompts is None:
                raise ValueError("no saved prompts; use the 'current mask' seed instead")
            session.seed_points_box(lc.obj_id, lc.target_prompts, lc.anchor_idx)
        for nb in self.neighbors:
            session.seed_points_box(nb["obj_id"], nb["prompts"], lc.anchor_idx)

    def run_ab(self):
        if not self.neighbors:
            print("[coprop] seed at least one neighbor first (the constraint is a no-op "
                  "with a single object)")
            return
        self._ensure_predictors()
        lc = self.lc
        mem = (self.variant == "memory")

        print(f"[coprop] baseline pass (neighbors off), seed={self.target_seed}")
        with MultiObjectCopropSession(self.video_predictor, lc.frames_dir) as sa:
            self._seed_all(sa)
            sa.run_bidirectional()
            seg_a = sa.video_segments

        print(f"[coprop] treatment pass (variant={self.variant})")
        with MultiObjectCopropSession(self.video_predictor, lc.frames_dir,
                                      non_overlap=not mem, non_overlap_mem_enc=mem) as sb:
            self._seed_all(sb)
            sb.run_bidirectional()
            seg_b = sb.video_segments

        neigh_ids = [nb["obj_id"] for nb in self.neighbors]
        alone = label_stack(seg_a, [lc.obj_id], lc.n_frames, lc.hw)
        withn = label_stack(seg_b, [lc.obj_id], lc.n_frames, lc.hw)
        neighbors = label_stack(seg_b, neigh_ids, lc.n_frames, lc.hw)
        diff = build_diff_stack(alone, withn, lc.obj_id)

        s = self._em_world
        lscale = (1.0, s, s)
        self._upsert_labels("target (alone)", alone, lscale)
        self._upsert_labels("target (w/ neighbors)", withn, lscale)
        self._upsert_labels("neighbors (propagated)", neighbors, lscale)
        self._upsert_labels("diff (1=lost, 2=gained)", diff, lscale)

        lost, gained = int((diff == 1).sum()), int((diff == 2).sum())
        print(f"[coprop] diff: lost {lost} px, gained {gained} px "
              f"(variant={self.variant}; output-only should gain 0)")

    def _upsert_labels(self, name, data, scale):
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name, opacity=0.5, scale=scale)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Standalone co-propagation A/B lab (no saving).")
    p.add_argument("--neuron", required=True)
    p.add_argument("--chain", type=int, required=True)
    from sam2_utils import config
    default_root = str(config.GT_PRED_DIR / "batch_masks_multichain")
    p.add_argument("--root", default=default_root,
                   help="output root holding <neuron>/chain_NN (default: multichain GT eval output)")
    args = p.parse_args()
    CopropLab(args.root, args.neuron, args.chain).run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module still imports torch-free and the helper tests still pass**

Run: `py -3 -m pytest tests/test_coprop_lab.py -v`
Expected: all tests pass (the new napari/magicgui code is import-lazy, so importing `coprop_lab` for the pure helpers must not pull in torch or napari). If an import error appears, move the offending import inside the function that uses it.

- [ ] **Step 3: Smoke-test the full app on a real chain (needs GPU + display)**

```bash
py -3 coprop_lab.py --neuron AVAL --chain 7 --root data/groundtruth/pred_p280/batch_masks_multichain
```
Expected: napari opens on the anchor frame with the EM, the preloaded target paint, and the dock. Clicking a neighbor cell shows its mask immediately. "run A/B" with one neighbor and the Test 1 preset prints `gained 0 px` and adds the four result layers; toggling `target (alone)` vs `target (w/ neighbors)` and viewing `diff` shows where the neighbor carved bleed. Switching to Test 2 and re-running shows the memory-variant trajectory change. Adjust `--root`/`--neuron`/`--chain` to a chain that exists.

- [ ] **Step 4: Final lint and full test gate, then commit**

```bash
ruff check coprop_lab.py tests/test_coprop_lab.py
py -3 -m pytest
git add coprop_lab.py
git commit -m "coprop-lab: napari viewer, click-to-seed neighbors, A/B run + diff layers + CLI"
```

---

## Self-review

**Spec coverage:**
- Purpose, disposability, no saving: Task 5 (`main`, no persistence). Done.
- Two tests to two flags: Task 3 (flag toggles), Task 5 (presets + variant). Done.
- Test 1 sanity check (output-only gains 0): Task 1 docstring + Task 5 readout. Done.
- Verified API facts (box needs `clear_old_points`, squeeze `(1,H,W)`, no-op with one object): Task 3 implementation and comments; Task 5 run guard. Done.
- Chain loader: Task 2. Session: Task 3. Neighbor click seeding with immediate preview: Tasks 4 and 5. Viewer app with the four result layers and readout: Task 5. Done.
- Tests for the pure helpers ported: Task 1. Done.
- Run command and tier-2 preference: Task 5 smoke + spec (no code needed). Done.

**Placeholder scan:** No TBD/TODO. The one conditional note (jpeg naming in Task 2 Step 1) gives an exact resolution path (inspect `_load_frame` and a real `frames_dir`), not a deferral.

**Type consistency:** `label_stack(segments, obj_ids, t, hw)` and `build_diff_stack(alone, withn, obj_id)` are used with matching signatures in Task 5. Session methods `seed_points_box` / `seed_mask` / `run_bidirectional` and the `video_segments` attribute match between Task 3 and Task 5. `predict_neighbor_at(...)` returns `(mask, prompts)` and is consumed that way in Task 5. `LabChain` fields used in Task 5 (`em`, `frames_dir`, `anchor_idx`, `obj_id`, `n_frames`, `hw`, `anchor_mask`, `target_prompts`) all exist in the Task 2 dataclass.
