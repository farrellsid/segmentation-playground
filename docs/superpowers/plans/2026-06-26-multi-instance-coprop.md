# Multi-instance co-propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gated `gui.py` action that co-propagates a chain's CATMAID neighbors as extra SAM2 objects and shows, side by side, the target mask with the non-overlap constraint OFF (a matched target-alone control) vs ON, to test the PI's neighbor-competition hypothesis on the target worm.

**Architecture:** Three additive layers. A pure `neighbor_chains(...)` finder in `pipeline/predict.py` (torch-free, unit-tested). A `MultiObjectPropagationSession` in `pipeline/propagate.py` that seeds N objects in one `inference_state`, toggles SAM2's non-overlap flags with set-and-restore, and propagates them together. A gated action in `gui.py` that wires the two together and adds new napari layers without touching any existing flow. The existing single-object `PropagationSession` and `run_chain` are not modified, so the production path stays byte-identical.

**Tech Stack:** Python 3.13 (`py -3`), numpy, pandas, pytest (CPU-only), SAM2 (`sam2.sam2_video_predictor`), napari + magicgui (GUI, not unit-tested).

## Global Constraints

- No em dashes anywhere (code, comments, docstrings, commit messages). Use commas, colons, parentheses, or separate sentences.
- Run the `humanizer` skill on any prose committed (docstrings, comments, commit messages) so committed text does not read as AI-generated.
- Tests are CPU-only and torch-free for pure logic: `py -3 -m pytest`. New pure-logic tests must not import torch.
- Lint with `ruff check .`. Clean only the files you touch; do not reformat the whole tree.
- The library (`pipeline.py`, `sam2_utils/`) must never import the drivers (`batch`, `gui`, `run_aval`) or `eval`. `tests/test_import_direction.py` enforces this.
- Commit incrementally, one concern per commit.
- Coordinate spaces carry a suffix (`_tif`, `_sam`, `_crop`/`_pcrop`). Keep neighbor coordinates in the same space as the target's propagation frames.
- The repo is a git repo on branch `repo-reorg`. Run all commands from `d:/Zhen Lab/SAM2 Segmentation/segmentation-playground`.

---

### Task 1: `neighbor_chains(...)` pure finder

Find the k nearest other chains that have a node inside the target chain's propagation window on a shared z-slice. Pure and torch-free, so it is unit-tested under the CPU-only rule.

**Files:**
- Modify: `pipeline/predict.py` (add `neighbor_chains` near `build_prompts`, around line 82)
- Modify: `pipeline/__init__.py` (export `neighbor_chains`)
- Test: `tests/test_neighbor_chains.py` (create)

**Interfaces:**
- Consumes: `annotate_df` columns `node_id`, `cell_name`, `z`, `x_tif`, `y_tif` (strings/numerics as elsewhere); `chains` list of `{cell_name, nodes:[node_id,...]}` (from `chains.json`); `sam2_utils.alignment.CropWindow` (optional, tier-2).
- Produces:
  ```python
  def neighbor_chains(
      target_chain: dict,
      annotate_df: "pd.DataFrame",
      chains: list[dict],
      *,
      scale: int,
      k: int = 3,
      crop_window: "Optional[alignment.CropWindow]" = None,
      frame_hw_sam: "Optional[tuple[int, int]]" = None,
  ) -> list[dict]:
      """Returns up to k dicts: {"chain": <chain dict>, "chain_idx": int,
      "cell_name": str, "min_dist_sam": float, "anchor_node_id": int,
      "anchor_catmaid_z": int}, nearest first."""
  ```
  `chain_idx` is the index of the neighbor chain within `chains`. `anchor_node_id` is the neighbor's in-window node nearest (in _sam) to any target node on a shared z; `anchor_catmaid_z` is that node's z.

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for pipeline.neighbor_chains (the CATMAID neighbor finder).

Torch-free and data-free like test_anchor_select: pipeline imports torch only
lazily, so the pure neighbor-selection logic needs no GPU and no EM stack.

Run either way:
    py -3 -m pytest tests/test_neighbor_chains.py
    py -3 tests/test_neighbor_chains.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import alignment


def _df(rows):
    """rows: list of (node_id, cell_name, z, x_tif, y_tif). x/y duplicated into catmaid x/y."""
    return pd.DataFrame(
        [{"node_id": n, "cell_name": c, "z": z, "x_tif": x, "y_tif": y, "x": x, "y": y}
         for (n, c, z, x, y) in rows]
    )


def test_picks_nearest_other_chains_and_excludes_target():
    # target chain A: nodes 1,2 at z0/z1 near x=100. Neighbors B (x=110, close),
    # C (x=130, farther), D (x=900, far). Same-z so all can contend.
    df = _df([
        (1, "A", 0, 100, 100), (2, "A", 1, 100, 100),
        (3, "B", 0, 110, 100),
        (4, "C", 0, 130, 100),
        (5, "D", 0, 900, 100),
    ])
    chains = [
        {"cell_name": "A", "nodes": [1, 2]},   # idx 0 (target)
        {"cell_name": "B", "nodes": [3]},       # idx 1
        {"cell_name": "C", "nodes": [4]},       # idx 2
        {"cell_name": "D", "nodes": [5]},       # idx 3
    ]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=2,
                                   frame_hw_sam=(1000, 1000))
    assert [o["cell_name"] for o in out] == ["B", "C"]   # nearest two, target excluded
    assert out[0]["chain_idx"] == 1
    assert out[0]["min_dist_sam"] < out[1]["min_dist_sam"]
    assert out[0]["anchor_node_id"] == 3


def test_drops_chains_with_no_in_window_node():
    # target window is the _sam frame 0..50. Neighbor B sits at x=110, outside it.
    df = _df([
        (1, "A", 0, 10, 10),
        (3, "B", 0, 110, 10),
    ])
    chains = [{"cell_name": "A", "nodes": [1]}, {"cell_name": "B", "nodes": [3]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3,
                                   frame_hw_sam=(50, 50))
    assert out == []                              # B is outside the 50x50 frame


def test_requires_shared_z_slice():
    # B only exists on z=5, target only on z=0: cannot contend, so dropped.
    df = _df([(1, "A", 0, 100, 100), (3, "B", 5, 101, 100)])
    chains = [{"cell_name": "A", "nodes": [1]}, {"cell_name": "B", "nodes": [3]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3,
                                   frame_hw_sam=(1000, 1000))
    assert out == []


def test_crop_window_maps_into_pcrop_and_filters():
    # tier-2: a 200x200 _tif window at origin (50,50), crop_scale 1, sam_scale 1.
    # target node at tif (100,100) -> in window. Neighbor at tif (120,120) -> in window;
    # neighbor at tif (300,300) -> outside window, dropped.
    cw = alignment.CropWindow(origin_tif=(50.0, 50.0), size_tif=(200, 200),
                              crop_scale=1, sam_scale=1)
    df = _df([
        (1, "A", 0, 100, 100),
        (3, "B", 0, 120, 120),
        (4, "C", 0, 300, 300),
    ])
    chains = [{"cell_name": "A", "nodes": [1]},
              {"cell_name": "B", "nodes": [3]},
              {"cell_name": "C", "nodes": [4]}]
    out = pipeline.neighbor_chains(chains[0], df, chains, scale=1, k=3, crop_window=cw)
    assert [o["cell_name"] for o in out] == ["B"]    # C is outside the crop window


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_neighbor_chains.py -v`
Expected: FAIL with `AttributeError: module 'pipeline' has no attribute 'neighbor_chains'`

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/predict.py` (after `build_prompts`, before `_point_in_mask`):

```python
def neighbor_chains(target_chain, annotate_df, chains, *, scale,
                    k=3, crop_window=None, frame_hw_sam=None):
    """The k nearest OTHER chains with a node inside the target's propagation window
    on a shared z-slice, nearest first. Pure (no torch, no SAM2): the seeds are built
    later by the GUI via the normal anchor path.

    A neighbor can only contend with the target where both have foreground on the same
    slice, so a chain is kept only if it has a node (a) on a z the target also occupies
    and (b) inside the propagation frame. Distance is measured in _sam px between the
    neighbor node and the nearest target node on that shared z.

    Window: tier-2 passes `crop_window` (an alignment.CropWindow), so nodes map _tif ->
    _pcrop and the window is the crop extent. The _sam path passes `frame_hw_sam` (the
    (H, W) of the scale-`scale` frame), so nodes map _tif -> _sam by /scale and the
    window is the whole frame. Exactly one of the two should be given.

    Returns up to k dicts: {chain, chain_idx, cell_name, min_dist_sam, anchor_node_id,
    anchor_catmaid_z}. anchor_node_id is the in-window neighbor node closest (in _sam) to
    a target node on a shared z; that node's z is anchor_catmaid_z.
    """
    from sam2_utils import alignment

    target_name = target_chain.get("cell_name")
    target_node_ids = {str(n) for n in target_chain["nodes"]}

    # target nodes -> {z: [(x_sam, y_sam), ...]}, in the propagation space.
    def _to_space(xy_tif):
        if crop_window is not None:
            return crop_window.tif_to_crop(xy_tif)        # _tif -> _pcrop
        return alignment.tif_to_sam(xy_tif, scale)        # _tif -> _sam

    def _in_window(xy_space):
        x, y = float(xy_space[0]), float(xy_space[1])
        if crop_window is not None:
            h, w = crop_window.crop_hw
        else:
            h, w = (int(frame_hw_sam[0]), int(frame_hw_sam[1]))
        return (0 <= x < w) and (0 <= y < h)

    tdf = annotate_df[annotate_df["node_id"].astype(str).isin(target_node_ids)]
    target_by_z: dict[int, list] = {}
    for _, r in tdf.iterrows():
        xy = _to_space((float(r["x_tif"]), float(r["y_tif"])))
        target_by_z.setdefault(int(r["z"]), []).append((float(xy[0]), float(xy[1])))
    target_zs = set(target_by_z)

    out = []
    for idx, ch in enumerate(chains):
        if ch is target_chain or ch.get("cell_name") == target_name:
            continue                                       # skip the target neuron
        cdf = annotate_df[annotate_df["node_id"].astype(str).isin({str(n) for n in ch["nodes"]})]
        best = None                                        # (dist, node_id, z)
        for _, r in cdf.iterrows():
            z = int(r["z"])
            if z not in target_zs:
                continue                                   # no shared slice -> cannot contend
            xy = _to_space((float(r["x_tif"]), float(r["y_tif"])))
            if not _in_window(xy):
                continue                                   # outside the propagation frame
            for (tx, ty) in target_by_z[z]:
                d = float(np.hypot(xy[0] - tx, xy[1] - ty))
                if best is None or d < best[0]:
                    best = (d, int(r["node_id"]), z)
        if best is not None:
            out.append({"chain": ch, "chain_idx": idx, "cell_name": ch.get("cell_name"),
                        "min_dist_sam": best[0], "anchor_node_id": best[1],
                        "anchor_catmaid_z": best[2]})

    out.sort(key=lambda o: o["min_dist_sam"])
    return out[:k]
```

Add the export to `pipeline/__init__.py`: in the `from .predict import (...)` block add `neighbor_chains,` (next to `build_prompts`), and add `"neighbor_chains",` to `__all__` next to `"build_prompts"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_neighbor_chains.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint and commit**

Run: `ruff check pipeline/predict.py pipeline/__init__.py tests/test_neighbor_chains.py`
Expected: clean (no errors)

```bash
git add pipeline/predict.py pipeline/__init__.py tests/test_neighbor_chains.py
git commit -m "pipeline: neighbor_chains finder (nearest in-window chains on shared z)"
```

---

### Task 2: `MultiObjectPropagationSession`

One `inference_state`, N seeded objects, non-overlap flags toggled with set-and-restore, propagate all objects together. The existing `PropagationSession` is not touched.

**Files:**
- Modify: `pipeline/propagate.py` (add the class after `PropagationSession`, before the module-level `propagate`)
- Modify: `pipeline/__init__.py` (export `MultiObjectPropagationSession`)
- Test: `tests/test_multiobj_session.py` (create) - tests only the torch-free flag set-and-restore via a fake predictor.

**Interfaces:**
- Consumes: a SAM2 video predictor with `init_state`, `reset_state`, `add_new_points_or_box`, `add_new_mask`, `propagate_in_video`, and the settable attributes `non_overlap_masks` (on `SAM2VideoPredictor`) and `non_overlap_masks_for_mem_enc` (on `SAM2Base`); `pipeline.Prompts`.
- Produces:
  ```python
  class MultiObjectPropagationSession:
      def __init__(self, video_predictor, frames_dir: str, *,
                   non_overlap: bool = False, non_overlap_mem_enc: bool = False,
                   offload_video_to_cpu: bool = True): ...
      def seed(self, obj_id: int, prompts: "Prompts", anchor_frame_idx: int, *,
               seed_box: bool = True, seed_points: bool = True,
               seed_negatives: bool = False) -> None: ...
      def run_bidirectional(self) -> None: ...   # forward then reverse over ALL seeded objects
      def close(self) -> None: ...               # restores both predictor flags; idempotent
      # accumulator, frame_idx-keyed, per object:
      self.video_segments: dict[int, dict[int, np.ndarray]]   # {frame_idx: {obj_id: bool mask}}
  ```
  Context-manager (`__enter__`/`__exit__`) like `PropagationSession`.

- [ ] **Step 1: Write the failing test**

```python
"""Unit test for MultiObjectPropagationSession's flag set-and-restore.

Torch-free: we drive the session with a fake predictor that records attribute
writes, so we can assert the non-overlap flags are set on enter and RESTORED on
close (the 'do not perturb a shared predictor' contract). The propagation math
itself is torch-bound and exercised manually in the GUI, not here.

Run either way:
    py -3 -m pytest tests/test_multiobj_session.py -v
    py -3 tests/test_multiobj_session.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline


class _FakePredictor:
    """Records the seed calls and lets us read the non-overlap flags."""
    def __init__(self, non_overlap_masks=False, non_overlap_masks_for_mem_enc=False):
        self.non_overlap_masks = non_overlap_masks
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.seeds = []

    def init_state(self, **kw):
        return {"state": True}

    def reset_state(self, state):
        pass

    def add_new_points_or_box(self, **kw):
        self.seeds.append(kw)

    def add_new_mask(self, **kw):
        self.seeds.append(kw)

    # never called in this test (no propagate), present for completeness
    def propagate_in_video(self, state, **kw):
        return iter(())


def test_flags_set_on_enter_and_restored_on_close():
    vp = _FakePredictor(non_overlap_masks=False, non_overlap_masks_for_mem_enc=False)
    sess = pipeline.MultiObjectPropagationSession(
        vp, "frames", non_overlap=True, non_overlap_mem_enc=True)
    # set while the session is live
    assert vp.non_overlap_masks is True
    assert vp.non_overlap_masks_for_mem_enc is True
    sess.close()
    # restored to the original values
    assert vp.non_overlap_masks is False
    assert vp.non_overlap_masks_for_mem_enc is False


def test_close_is_idempotent_and_restores_once():
    vp = _FakePredictor(non_overlap_masks=True, non_overlap_masks_for_mem_enc=False)
    sess = pipeline.MultiObjectPropagationSession(vp, "frames", non_overlap=False)
    assert vp.non_overlap_masks is False     # forced off while live
    sess.close()
    sess.close()                              # second close is a no-op
    assert vp.non_overlap_masks is True       # restored to original


def test_seed_sends_one_call_per_object_with_obj_id():
    vp = _FakePredictor()
    import numpy as np
    with pipeline.MultiObjectPropagationSession(vp, "frames") as sess:
        p = pipeline.Prompts(points_sam=np.array([[5.0, 5.0]]), labels=np.array([1]),
                             box_sam=np.array([1.0, 1.0, 9.0, 9.0], dtype=np.float32))
        sess.seed(1, p, 0)
        sess.seed(2, p, 0)
    obj_ids = [s["obj_id"] for s in vp.seeds]
    assert obj_ids == [1, 2]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_multiobj_session.py -v`
Expected: FAIL with `AttributeError: module 'pipeline' has no attribute 'MultiObjectPropagationSession'`

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/propagate.py` after the `PropagationSession` class (after line 251, before the module-level `def propagate`):

```python
class MultiObjectPropagationSession:
    """Co-propagate SEVERAL objects in ONE inference_state to test the neighbor effect.

    SAM2 tracks each object's memory independently; the only coupling is the per-pixel
    non-overlap argmax (_apply_non_overlapping_constraints). This session seeds N objects
    (a target + its neighbors) and toggles that argmax so the OFF run is a matched
    target-alone control and the ON run is the treatment, the only difference being the
    constraint.

    Two SAM2 flags, both plain attributes on the predictor (so we set them at runtime and
    RESTORE them on close, never leaving a shared predictor mutated, the same discipline as
    the IoU hook):
      non_overlap            -> video_predictor.non_overlap_masks (OUTPUT masks only)
      non_overlap_mem_enc    -> video_predictor.non_overlap_masks_for_mem_enc (fed back into
                                memory, so it changes the propagation trajectory; the variant
                                that can actually improve tracking).

    Single-object PropagationSession is unchanged; this is a separate, additive path.
    """

    def __init__(self, video_predictor, frames_dir: str, *,
                 non_overlap: bool = False, non_overlap_mem_enc: bool = False,
                 offload_video_to_cpu: bool = True):
        self.vp = video_predictor
        self.obj_ids: list[int] = []
        # save the originals so close() can restore them.
        self._orig_no = getattr(video_predictor, "non_overlap_masks", False)
        self._orig_no_mem = getattr(video_predictor, "non_overlap_masks_for_mem_enc", False)
        video_predictor.non_overlap_masks = bool(non_overlap)
        video_predictor.non_overlap_masks_for_mem_enc = bool(non_overlap_mem_enc)

        self.inference_state = video_predictor.init_state(
            video_path=frames_dir, offload_video_to_cpu=offload_video_to_cpu)
        video_predictor.reset_state(self.inference_state)

        self.video_segments: dict[int, dict[int, np.ndarray]] = {}
        self._closed = False

    def seed(self, obj_id: int, prompts: Prompts, anchor_frame_idx: int, *,
             seed_box: bool = True, seed_points: bool = True,
             seed_negatives: bool = False) -> None:
        """Seed one object's anchor frame (box + positive point by default), same prompt
        handling as PropagationSession.seed but for an explicit obj_id."""
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
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state, frame_idx=int(anchor_frame_idx),
            obj_id=int(obj_id), box=box, points=pts, labels=labels)
        if obj_id not in self.obj_ids:
            self.obj_ids.append(int(obj_id))

    def _drain(self, *, reverse: bool) -> None:
        for f, obj_ids, mask_logits in self.vp.propagate_in_video(
                self.inference_state, reverse=reverse):
            fi = int(f)
            per_obj = self.video_segments.setdefault(fi, {})
            for i, oid in enumerate(obj_ids):
                per_obj[int(oid)] = (mask_logits[i].cpu().numpy() > 0.0)

    def run_bidirectional(self) -> None:
        """Forward then reverse over ALL seeded objects (one shared memory)."""
        self._drain(reverse=False)
        self._drain(reverse=True)

    def close(self) -> None:
        """Restore both predictor flags. Idempotent."""
        if not self._closed:
            self.vp.non_overlap_masks = self._orig_no
            self.vp.non_overlap_masks_for_mem_enc = self._orig_no_mem
            self._closed = True

    def __enter__(self) -> "MultiObjectPropagationSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

Add the export to `pipeline/__init__.py`: change line 99 to
`from .propagate import FrameResult, MultiObjectPropagationSession, PropagationSession, propagate, _attach_iou_hook`
and add `"MultiObjectPropagationSession",` to `__all__` next to `"PropagationSession"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_multiobj_session.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite + lint**

Run: `py -3 -m pytest -q && ruff check pipeline/propagate.py pipeline/__init__.py tests/test_multiobj_session.py`
Expected: all tests pass, ruff clean

- [ ] **Step 6: Commit**

```bash
git add pipeline/propagate.py pipeline/__init__.py tests/test_multiobj_session.py
git commit -m "pipeline: MultiObjectPropagationSession (co-propagate N objects, non-overlap set/restore)"
```

---

### Task 3: Neighbor seed builder (full anchor path)

A helper the GUI calls to turn a `neighbor_chains` entry into a seeded `Prompts` (box + point) in the target's propagation space, via the same anchor path the target uses. Lives in `gui.py` (it touches the image predictor and is GUI-only orchestration; keeping it out of the library preserves the import-direction rule and the torch-free test rule).

**Files:**
- Modify: `gui.py` (add a method `_seed_neighbor(...)` on `ReviewGUI`, near `_ensure_session` around line 1028)

**Interfaces:**
- Consumes: `pipeline.build_prompts`, `pipeline.image_predict`, `pipeline.anchor_crop_predict`, `pipeline.box_from_mask`, `pipeline.load_frame_sam`, `pipeline.Prompts`; `self.ctx.annotate_df`, `self.ctx.cfg`, `self.ctx.image_predictor`, `self._cw` (CropWindow or None), `self.data.frame_to_z`.
- Produces: `_seed_neighbor(self, nb: dict) -> Optional[tuple[int, Prompts, int]]` returning `(anchor_frame_idx, prompts_in_propagation_space, neighbor_obj_id)`, or None if the neighbor mask comes back empty. The neighbor's `obj_id` is chosen by the caller (Task 4) and passed in via `nb["obj_id"]`.

- [ ] **Step 1: Add the method (no separate unit test: torch-bound, exercised in Task 4's manual run)**

Add to `gui.py` in the `ReviewGUI` class (right after `_ensure_session`, around line 1034):

```python
    def _seed_neighbor(self, nb: dict):
        """Build a (box + point) seed for one neighbor chain in THIS chain's propagation
        space, via the same anchor path the target uses. Returns
        (anchor_frame_idx, Prompts, obj_id) or None if the neighbor mask is empty.

        nb is a pipeline.neighbor_chains entry plus an injected "obj_id". The neighbor is
        seeded on its own in-window anchor node/z (nb["anchor_node_id"] / ["anchor_catmaid_z"]),
        mapped onto the chain's frame index via self.data.frame_to_z."""
        cfg = self.ctx.cfg
        z = int(nb["anchor_catmaid_z"])
        z_to_frame = {int(zz): int(fi) for fi, zz in self.data.frame_to_z.items()}
        if z not in z_to_frame:
            return None                          # neighbor's anchor z is not a frame we propagate
        frame_idx = z_to_frame[z]

        prompts = pipeline.build_prompts(
            nb["anchor_node_id"], z, self.ctx.annotate_df, scale=cfg.scale,
            k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)

        if self._cw is not None:                 # tier-2: predict in the SAME _pcrop window
            image_full, full_hw = pipeline.load_frame_sam(z, scale=1)
            mask, _score, _cw, prompts_anchor = pipeline.anchor_crop_predict(
                self.ctx.image_predictor, image_full, full_hw, nb["anchor_node_id"],
                prompts, self.ctx.annotate_df, scale=cfg.scale,
                crop_size_tif=cfg.crop_size_tif, crop_scale=self._cw.crop_scale, cw=self._cw,
                multimask=cfg.multimask_anchor)
            box = pipeline.box_from_mask(mask, margin=cfg.box_margin,
                                         image_hw_sam=mask.shape[:2])
            if box is None:
                return None
            seed = pipeline.Prompts(
                points_sam=np.asarray(prompts_anchor.points_sam, dtype=float),
                labels=np.asarray(prompts_anchor.labels, dtype=int),
                box_sam=np.asarray(box, dtype=np.float32))
        else:                                    # _sam path
            image_sam, _full_hw = pipeline.load_frame_sam(z, scale=cfg.scale)
            mask, _score, _logits = pipeline.image_predict(
                self.ctx.image_predictor, image_sam, prompts, multimask=cfg.multimask_anchor)
            box = pipeline.box_from_mask(mask, margin=cfg.box_margin,
                                         image_hw_sam=mask.shape[:2])
            if box is None:
                return None
            prompts.box_sam = np.asarray(box, dtype=np.float32)
            seed = prompts
        return frame_idx, seed, int(nb["obj_id"])
```

- [ ] **Step 2: Sanity-check the import surface**

Run: `py -3 -c "import pipeline; print(all(hasattr(pipeline, n) for n in ['build_prompts','image_predict','anchor_crop_predict','box_from_mask','load_frame_sam','Prompts','neighbor_chains','MultiObjectPropagationSession']))"`
Expected: `True`

- [ ] **Step 3: Lint and commit**

Run: `ruff check gui.py`
Expected: clean

```bash
git add gui.py
git commit -m "gui: _seed_neighbor builds a box+point neighbor seed via the anchor path"
```

---

### Task 4: Gated "co-propagate with neighbors" action + overlay layers

The user-facing action. Finds neighbors (selectable k, manual add/remove), runs the multi-object session twice (OFF then ON) over the open chain's frames, and adds new napari layers without touching the existing `mask` layer or any existing flow.

**Files:**
- Modify: `gui.py` (add `coprop_neighbors(...)` and its dock widgets/key; reuse `_seed_neighbor`, `neighbor_chains`, `MultiObjectPropagationSession`)

**Interfaces:**
- Consumes: `pipeline.neighbor_chains`, `pipeline.MultiObjectPropagationSession`, `self._seed_neighbor`, `self.data` (frames_dir, frame_to_z, obj_id, anchor_idx), `self._cw`, `self._lscale`, `self.viewer`, `self.ctx`.
- Produces: new napari layers named `mask (alone)`, `mask (w/ neighbors)`, `neighbors`, `diff`; a `neighbor count (k)` spin box and a `co-propagate neighbors (M)` button in the dock; a list widget of candidate neighbor chains with checkboxes for manual add/remove. No change to existing layers or actions.

- [ ] **Step 1: Add the widgets**

The dock is a single `Container(widgets=[...], labels=True)` built inline near the end of `_build_widgets` (gui.py:1132) and wrapped in a `QScrollArea`. There is no persistent container to `extend`; widgets are created in `_build_widgets` and listed in `widgets=[...]`.

First, add `Select` to the magicgui import at the top of `_build_widgets` (gui.py:1041-1042). Change:
```python
        from magicgui.widgets import (Container, PushButton, ComboBox, Label, LineEdit,
                                      FloatSpinBox, SpinBox, CheckBox)
```
to add `Select`:
```python
        from magicgui.widgets import (Container, PushButton, ComboBox, Label, LineEdit,
                                      FloatSpinBox, SpinBox, CheckBox, Select)
```

Then, alongside the recrop controls (after the `recrop`/`pick_region` block, around gui.py:1098), create the widgets:
```python
        # co-propagate with CATMAID neighbors (the neighbor-competition experiment).
        self._k_spin = SpinBox(label="neighbor count (k)", value=3, min=0, max=12)
        self._neighbor_select = Select(label="neighbors (override)", choices=[])
        coprop_btn = PushButton(text="⧉ co-propagate neighbors (M)")
        coprop_btn.changed.connect(self.coprop_neighbors)
```

Finally, add them to the `Container(widgets=[...])` list (gui.py:1132) with a separator label, right before the `, disposition,` group:
```python
            Label(value=", neighbors, "), self._k_spin, self._neighbor_select, coprop_btn,
```

- [ ] **Step 2: Bind the key**

In `_bind_keys` (around line 271), add alongside the other `@self.viewer.bind_key` handlers:

```python
        @self.viewer.bind_key("m", overwrite=True)
        def _m(_v):
            self.coprop_neighbors()
```

- [ ] **Step 3: Add the action**

Add to the `ReviewGUI` class (near `resume_propagation`):

```python
    def coprop_neighbors(self, *_) -> None:
        """Co-propagate the open chain WITH its CATMAID neighbors and overlay the target
        mask with the non-overlap constraint OFF (a matched target-alone control) vs ON.

        The only variable between the two runs is the constraint, so any difference in the
        target mask is the neighbor-competition effect. Adds new layers; the existing 'mask'
        layer and every existing flow are left untouched.
        """
        if self.data is None or self._recrop_picking:
            return
        self.ctx.ensure_predictors(need_image=True, need_video=True)

        target_obj = self.data.obj_id
        k = int(self._k_spin.value)
        frame_hw_sam = None
        if self._cw is None:
            any_mask = next(iter(self.data.video_segments.values()))[target_obj]
            any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
            frame_hw_sam = np.asarray(any_mask).shape

        # candidates: manual override (the Select) wins; else nearest-k auto.
        cands = pipeline.neighbor_chains(
            self.chain, self.ctx.annotate_df, self.ctx.chains, scale=self.ctx.cfg.scale,
            k=max(k, 12), crop_window=self._cw, frame_hw_sam=frame_hw_sam)
        self._neighbor_select.choices = [f"{c['chain_idx']}:{c['cell_name']}" for c in cands]
        chosen_keys = list(self._neighbor_select.value or [])
        if chosen_keys:
            cands = [c for c in cands if f"{c['chain_idx']}:{c['cell_name']}" in chosen_keys]
        else:
            cands = cands[:k]
        if not cands:
            print("[gui] no in-window neighbor chains found for this chain")
            return

        # assign neighbor obj_ids that do not collide with the target's.
        seeds = []
        next_id = max(target_obj + 1, 2)
        for c in cands:
            c = dict(c, obj_id=next_id)
            s = self._seed_neighbor(c)
            if s is not None:
                seeds.append(s)
                next_id += 1
        print(f"[gui] co-prop: target obj {target_obj} + {len(seeds)} neighbor(s) "
              f"{[c['cell_name'] for c in cands][:len(seeds)]}")

        # the target's own seed: reuse the chain's saved seed at its anchor frame.
        target_seed = (self.data.anchor_idx, self._state.prompts, target_obj)

        def _run(non_overlap_mem_enc: bool):
            sess = pipeline.MultiObjectPropagationSession(
                self.ctx.video_predictor, self.data.frames_dir,
                non_overlap=non_overlap_mem_enc, non_overlap_mem_enc=non_overlap_mem_enc)
            try:
                fi, pr, oid = target_seed
                sess.seed(oid, pr, int(fi))
                for (nfi, npr, noid) in seeds:
                    sess.seed(noid, npr, int(nfi))
                sess.run_bidirectional()
                return sess.video_segments
            finally:
                sess.close()

        off = _run(False)    # neighbors present but no coupling == target alone
        on = _run(True)      # treatment: non-overlap fed back into memory

        t = max(self.data.video_segments) + 1
        H, W = self._target_hw()
        alone = _label_stack_from_segments(off, self.data.frame_to_z, target_obj, t, (H, W))
        withn = _label_stack_from_segments(on, self.data.frame_to_z, target_obj, t, (H, W))
        nbr = self._neighbor_label_stack(on, [s[2] for s in seeds], t, (H, W))
        diff = self._diff_stack(alone, withn, target_obj, t, (H, W))

        s = self._lscale
        for name in ("mask (alone)", "mask (w/ neighbors)", "neighbors", "diff"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.viewer.add_labels(alone, name="mask (alone)", opacity=0.5, scale=s, visible=False)
        self.viewer.add_labels(withn, name="mask (w/ neighbors)", opacity=0.5, scale=s)
        self.viewer.add_labels(nbr, name="neighbors", opacity=0.4, scale=s)
        self.viewer.add_labels(diff, name="diff", opacity=0.7, scale=s)
        print("[gui] co-prop done. Toggle 'mask (alone)' vs 'mask (w/ neighbors)'; "
              "'diff' shows target pixels lost (1) and gained (2) under the constraint. "
              "Screenshot to document. Nothing on disk was changed.")
```

Add the three small helpers as methods on `ReviewGUI` (they read `self.data`, so they are methods, not the module-level `_label_stack_from_segments`):

```python
    def _target_hw(self):
        any_mask = next(iter(self.data.video_segments.values()))[self.data.obj_id]
        any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
        return np.asarray(any_mask).shape

    def _neighbor_label_stack(self, segments, neighbor_obj_ids, t, hw):
        """(T,H,W) uint8: each neighbor obj_id painted with its own label so they show in
        distinct colors. Background 0."""
        H, W = hw
        out = np.zeros((t, H, W), dtype=np.uint8)
        for fi, seg in segments.items():
            if not (0 <= fi < t):
                continue
            for oid in neighbor_obj_ids:
                if oid in seg:
                    out[fi][np.asarray(seg[oid]).astype(bool)] = oid
        return out

    def _diff_stack(self, alone, withn, obj_id, t, hw):
        """(T,H,W) uint8: 1 where the target had a pixel ALONE but lost it under the
        constraint (bleed carved out), 2 where it GAINED one. The visual read of the test."""
        a = (alone == obj_id)
        b = (withn == obj_id)
        out = np.zeros_like(alone, dtype=np.uint8)
        out[a & ~b] = 1     # lost (carved-out bleed)
        out[b & ~a] = 2     # gained
        return out
```

- [ ] **Step 4: Smoke-test the action manually (GUI, target worm)**

This is the manual verification (the session is torch-bound, no CPU test). With the SAM2 checkpoints and the target-worm data available:

Run: `py -3 gui.py` (or the repo's documented GUI entry point; see `docs/how-to` for the neuron-review/per-chain launch)
Then: open a chain known to bleed, set k=3, press `M`.
Expected: console prints the target + neighbor cell names, then "co-prop done"; four new layers appear; toggling `mask (alone)` vs `mask (w/ neighbors)` shows whether the target sheds bleed; `diff` highlights lost/gained pixels. Confirm the original `mask` layer and the saved files are unchanged (no write happened).

- [ ] **Step 5: Run the full suite + lint**

Run: `py -3 -m pytest -q && ruff check gui.py`
Expected: tests pass (the new GUI code is import-clean; no new CPU tests), ruff clean

- [ ] **Step 6: Commit**

```bash
git add gui.py
git commit -m "gui: co-propagate-with-neighbors action + OFF/ON overlay layers (M)"
```

---

### Task 5: Documentation

Record the new action and the experiment so the docs do not rot (per the repo's "where to record what" table).

**Files:**
- Modify: `docs/reference/cli.md` or the GUI how-to (whichever documents `gui.py` keys; check `docs/reference/code-map.md` for the GUI doc home)
- Modify: `docs/CHANGELOG.md` (add a dated entry)

- [ ] **Step 1: Document the GUI action**

In the per-chain GUI how-to, add the `M` key / "co-propagate neighbors" action: what it does (OFF/ON non-overlap A/B against CATMAID neighbors), the `k` knob and manual override, the four layers it adds, and that it writes nothing to disk. Keep it present-tense and run the `humanizer` skill on the prose.

- [ ] **Step 2: Add a CHANGELOG entry**

Append a `2026-06` entry summarizing the multi-instance co-propagation experiment hook (mechanism, OFF/ON A/B, the new session + finder + GUI action), linking the spec at `docs/superpowers/specs/2026-06-26-multi-instance-coprop-design.md`. Humanize the prose; no em dashes.

- [ ] **Step 3: Verify links and dashes**

Run: `grep -rnE "—|–" docs/CHANGELOG.md docs/reference/ | head` (expect no hits in the lines you added)
Confirm any new internal markdown links resolve.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: co-propagate-with-neighbors GUI action + changelog entry"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Task 1 = `neighbor_chains` finder; Task 2 = `MultiObjectPropagationSession` with non-overlap set/restore (both flag variants); Task 3 = full anchor-path neighbor seed; Task 4 = gated GUI action with selectable k + manual add/remove + OFF/ON overlay + diff; Task 5 = docs. The OFF run with neighbors seeded is the matched target-alone control, per the spec's A/B.
- **Non-goals honored:** no GT scoring, no batch driver, no change to `PropagationSession`/`run_chain`, no new GUI window.
- **Type consistency:** `neighbor_chains` returns dicts with `chain_idx`/`cell_name`/`anchor_node_id`/`anchor_catmaid_z`/`min_dist_sam`; Task 4 injects `obj_id` and Task 3 reads exactly those keys. `MultiObjectPropagationSession.seed(obj_id, prompts, anchor_frame_idx, ...)` matches the `(frame_idx, prompts, obj_id)` tuple order produced by `_seed_neighbor` (the caller unpacks and reorders explicitly).
- **Verification gotcha:** if `_state.prompts` is None for an opened chain (no state.json), Task 4 must fall back to seeding the target from `self._prompts_for_frame(self.data.anchor_idx)`; add that guard during Task 4 if the smoke test surfaces it.
- **Dock assembly:** Task 4 Step 1 assumes the existing dock is a magicgui `Container`. Before editing, read the actual `_build_widgets` assembly (around gui.py:210-265) and match its pattern (it may use a napari `add_dock_widget` with a composed `Container`); adjust the `extend`/layout call accordingly.
