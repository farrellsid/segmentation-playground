# Neuron-level review GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second review tool, `gui_neuron.py`, that lets the annotator review and correct a whole neuron at once, with its branches (chains) shown as one multi-color object on a single per-neuron crop canvas.

**Architecture:** Each branch stays a separate SAM2 object (today's per-chain tracking, unchanged). A per-neuron crop window (`_ncrop`), sized to the neuron's whole skeleton bbox at an adaptive `crop_scale`, is the one grid every branch remaps into, so mixed `_sam`/`_pcrop` branches share one space. One napari Labels layer holds an integer per branch; `selected_label` is the active branch. Single prompts/box layers are scoped to the active branch. `gui.py` and the batch are untouched.

**Tech Stack:** Python, numpy, OpenCV (cv2), pandas, napari + magicgui (GUI), SAM2 (torch, GPU, lazy). Tests are CPU-only and torch-free.

## Global Constraints

- No em dashes anywhere (code, comments, docs, commit messages). Run the `humanizer` skill on committed prose.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. New pure-logic tests must not import torch or napari.
- Lint with `ruff check .`; clean only files you touch. Do not reformat the tree.
- The library (`pipeline/`, `sam2_utils/`) must never import drivers (`batch`, `gui`, `gui_neuron`, `run_aval`, `eval`). `tests/test_import_direction.py` enforces this. New library code goes in `pipeline/`; new driver code goes in `gui_neuron.py`.
- `gui_neuron.py` is a driver; it may import from `gui.py` (driver to driver is allowed) and from `pipeline`/`sam2_utils`.
- Commit incrementally, one concern per commit. Use Windows shell (`py -3`).
- Run commands from the repo root: `d:\Zhen Lab\SAM2 Segmentation\segmentation-playground`.

---

## File structure

- `pipeline/crop.py` (modify): add `_neuron_skeleton_box_tif`, `neuron_crop_window`, `remap_mask_to_window`. Pure geometry, torch-free.
- `pipeline/__init__.py` (modify): export the three new names.
- `tests/test_neuron_crop.py` (create): unit tests for the three new pure functions.
- `gui_neuron.py` (create): the `NeuronReviewGUI` driver and a `launch` entry point. Imports `ReviewContext` and helpers from `gui.py`.
- `tests/test_neuron_view.py` (create): unit tests for the torch/napari-free helpers factored out of `gui_neuron.py` (`neurons_on_disk`, `build_neuron_label_volume`).
- `docs/how-to/review-flagged-chains.md` (modify) or a new `docs/how-to/review-a-neuron.md` (create): document the new tool.
- `docs/reference/code-map.md` (modify): add the `gui_neuron.py` entry.

---

## Task 1: Per-neuron crop window (`neuron_crop_window`)

**Files:**
- Modify: `pipeline/crop.py` (add after `node_crop_window`)
- Modify: `pipeline/__init__.py`
- Test: `tests/test_neuron_crop.py`

**Interfaces:**
- Consumes: `alignment.CropWindow.around_box`, `PipelineConfig` fields `chain_crop_pad_tif`, `chain_crop_scale`, `chain_crop_max_px`, `scale`.
- Produces:
  - `_neuron_skeleton_box_tif(chains: list[dict], annotate_df) -> tuple[float,float,float,float]` (xyxy `_tif`, union of all chains' node bboxes).
  - `neuron_crop_window(chains, annotate_df, *, cfg, image_hw_tif) -> alignment.CropWindow`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_neuron_crop.py
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pipeline
from pipeline import PipelineConfig


def _df():
    # two chains of one neuron; node xy_tif span x:1000..1600, y:1000..1400
    return pd.DataFrame({
        "node_id": ["a", "b", "c", "d"],
        "z": [1000, 1001, 1002, 1003],
        "x_tif": [1000.0, 1200.0, 1400.0, 1600.0],
        "y_tif": [1000.0, 1100.0, 1300.0, 1400.0],
    })


def test_neuron_skeleton_box_unions_all_chains():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b"]}, {"cell_name": "N", "nodes": ["c", "d"]}]
    box = pipeline._neuron_skeleton_box_tif(chains, df)
    assert box == (1000.0, 1000.0, 1600.0, 1400.0)


def test_neuron_crop_window_contains_skeleton_and_pads():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b"]}, {"cell_name": "N", "nodes": ["c", "d"]}]
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop_pad_tif=100, chain_crop_scale=2, chain_crop_max_px=100000)
    cw = pipeline.neuron_crop_window(chains, df, cfg=cfg, image_hw_tif=(8000, 8000))
    # skeleton bbox 600x400 + 100 pad/side -> 800x600, centered on (1300, 1200)
    assert cw.size_tif == (800, 600)
    assert cw.origin_tif == (900.0, 900.0)
    assert cw.crop_scale == 2


def test_neuron_crop_window_bumps_scale_over_max_px():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b", "c", "d"]}]
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop_pad_tif=0, chain_crop_scale=2, chain_crop_max_px=300)
    cw = pipeline.neuron_crop_window(chains, df, cfg=cfg, image_hw_tif=(8000, 8000))
    # longest edge 600 _tif / 300 -> ceil 2; max(2, 2) = 2
    assert cw.crop_scale == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3 -m pytest tests/test_neuron_crop.py -q`
Expected: FAIL (`module 'pipeline' has no attribute '_neuron_skeleton_box_tif'`).

- [ ] **Step 3: Implement the two functions**

In `pipeline/crop.py`, after `node_crop_window`:

```python
def _neuron_skeleton_box_tif(chains, annotate_df) -> tuple[float, float, float, float]:
    """Union (x0, y0, x1, y1) bbox in _tif px over the nodes of ALL a neuron's chains."""
    ids = set()
    for ch in chains:
        ids.update(str(n) for n in ch["nodes"])
    sub = annotate_df[annotate_df["node_id"].astype(str).isin(ids)]
    xs = sub["x_tif"].to_numpy(dtype=float)
    ys = sub["y_tif"].to_numpy(dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def neuron_crop_window(chains, annotate_df, *, cfg: "PipelineConfig",
                       image_hw_tif: tuple[int, int]) -> "alignment.CropWindow":
    """A single CropWindow covering a whole neuron's skeleton bbox (over all its chains),
    padded by cfg.chain_crop_pad_tif, at an adaptive crop_scale bumped coarser if the
    padded extent's longest edge would exceed cfg.chain_crop_max_px. The unified canvas
    for the neuron-review GUI: every branch (_sam or _pcrop) remaps into this one grid."""
    box_tif = _neuron_skeleton_box_tif(chains, annotate_df)
    H_tif, W_tif = int(image_hw_tif[0]), int(image_hw_tif[1])
    pad = int(cfg.chain_crop_pad_tif)
    cx = 0.5 * (box_tif[0] + box_tif[2])
    cy = 0.5 * (box_tif[1] + box_tif[3])
    w_tif = min(W_tif, (box_tif[2] - box_tif[0]) + 2 * pad)
    h_tif = min(H_tif, (box_tif[3] - box_tif[1]) + 2 * pad)
    exp_box = (cx - w_tif / 2.0, cy - h_tif / 2.0, cx + w_tif / 2.0, cy + h_tif / 2.0)
    longest = max(w_tif, h_tif, 1.0)
    crop_scale = max(int(cfg.chain_crop_scale),
                     int(np.ceil(longest / float(cfg.chain_crop_max_px))))
    return alignment.CropWindow.around_box(
        exp_box, pad_tif=0, image_hw_tif=image_hw_tif,
        crop_scale=crop_scale, sam_scale=cfg.scale)
```

In `pipeline/__init__.py`, add to the `from .crop import (...)` block and `__all__`:
`_neuron_skeleton_box_tif`, `neuron_crop_window`.

- [ ] **Step 4: Run to verify they pass**

Run: `py -3 -m pytest tests/test_neuron_crop.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/crop.py pipeline/__init__.py tests/test_neuron_crop.py
git commit -m "pipeline: neuron_crop_window (per-neuron crop over the whole skeleton bbox)"
```

---

## Task 2: Remap a branch mask into an arbitrary window (`remap_mask_to_window`)

**Files:**
- Modify: `pipeline/crop.py`
- Modify: `pipeline/__init__.py`
- Test: `tests/test_neuron_crop.py` (append)

**Interfaces:**
- Consumes: `alignment.CropWindow` (`origin_tif`, `size_tif`, `crop_scale`, `crop_hw`).
- Produces: `remap_mask_to_window(mask, *, src_origin_tif, src_size_tif, dst_cw) -> np.ndarray` returning a bool array of shape `dst_cw.crop_hw` with the mask resampled (nearest) and placed at its `_tif` footprint, clipped to the window.

Note: we add this alongside `chain_masks_in_sam` and do NOT refactor that function, to avoid disturbing the working aggregation path (YAGNI). The neuron GUI computes each branch's `src_origin_tif`/`src_size_tif` from its `CropWindow` (tier-2) or from the `_sam` full frame (legacy).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_neuron_crop.py
import cv2  # noqa: E402


def test_remap_mask_to_window_places_and_scales():
    from sam2_utils.alignment import CropWindow
    # dst window: tif origin (1000,1000), 400x400 tif, crop_scale 2 -> 200x200 grid
    dst = CropWindow(origin_tif=(1000.0, 1000.0), size_tif=(400, 400), crop_scale=2, sam_scale=8)
    # source mask: a 100x100 tif footprint at tif (1100,1100), given at crop_scale 1 (100x100 px)
    src = np.zeros((100, 100), dtype=bool)
    src[20:80, 20:80] = True
    out = pipeline.remap_mask_to_window(
        src, src_origin_tif=(1100.0, 1100.0), src_size_tif=(100, 100), dst_cw=dst)
    assert out.shape == (200, 200)              # dst.crop_hw
    # src footprint maps to dst crop coords: origin (1100-1000)/2 = 50, size 100/2 = 50
    ys, xs = np.where(out)
    assert 45 <= xs.min() <= 60 and 90 <= xs.max() <= 105   # ~[50, 100) in x
    assert 45 <= ys.min() <= 60 and 90 <= ys.max() <= 105


def test_remap_mask_to_window_clips_out_of_window():
    from sam2_utils.alignment import CropWindow
    dst = CropWindow(origin_tif=(1000.0, 1000.0), size_tif=(400, 400), crop_scale=2, sam_scale=8)
    src = np.ones((50, 50), dtype=bool)         # footprint entirely left of the window
    out = pipeline.remap_mask_to_window(
        src, src_origin_tif=(0.0, 0.0), src_size_tif=(50, 50), dst_cw=dst)
    assert out.shape == (200, 200) and not out.any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/test_neuron_crop.py -k remap -q`
Expected: FAIL (`has no attribute 'remap_mask_to_window'`).

- [ ] **Step 3: Implement**

In `pipeline/crop.py`:

```python
def remap_mask_to_window(mask, *, src_origin_tif, src_size_tif,
                         dst_cw: "alignment.CropWindow") -> np.ndarray:
    """Place a bool ``mask`` (in some source crop space) into ``dst_cw``'s grid.

    The mask occupies the _tif rectangle [src_origin_tif, src_origin_tif + src_size_tif].
    We resize it to that rectangle's size in dst crop px (src_size_tif / dst.crop_scale)
    and paste it at the rectangle's dst origin ((src_origin - dst_origin) / dst.crop_scale),
    clipped to the window. Nearest-neighbour, like chain_masks_in_sam. The general remap
    behind the neuron view (chain_masks_in_sam stays the dedicated native->_sam case)."""
    import cv2
    dh, dw = dst_cw.crop_hw
    out = np.zeros((dh, dw), dtype=bool)
    m = np.asarray(mask).astype(np.uint8)
    if not m.size:
        return out
    s = float(dst_cw.crop_scale)
    tw = max(1, int(round(src_size_tif[0] / s)))
    th = max(1, int(round(src_size_tif[1] / s)))
    rm = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
    x0 = int(round((src_origin_tif[0] - dst_cw.origin_tif[0]) / s))
    y0 = int(round((src_origin_tif[1] - dst_cw.origin_tif[1]) / s))
    # intersect [x0, x0+tw) x [y0, y0+th) with [0, dw) x [0, dh)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dw, x0 + tw), min(dh, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return out
    out[dy0:dy1, dx0:dx1] = rm[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    return out
```

Add `remap_mask_to_window` to `pipeline/__init__.py` import + `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `py -3 -m pytest tests/test_neuron_crop.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/crop.py pipeline/__init__.py tests/test_neuron_crop.py
git commit -m "pipeline: remap_mask_to_window (place a branch mask into an arbitrary crop grid)"
```

---

## Task 3: Neuron enumeration + label-volume builder (pure helpers)

**Files:**
- Create: `gui_neuron.py`
- Test: `tests/test_neuron_view.py`

**Interfaces:**
- Consumes: `sam2_utils.review_queue.ReviewQueue.all_chains`.
- Produces (module-level, torch/napari-free):
  - `neurons_on_disk(output_root) -> list[tuple[str, list[int]]]`: `(neuron, [chain_idx,...])` for every neuron with on-disk chains, sorted.
  - `build_neuron_label_volume(branch_masks: dict[int, dict[int, np.ndarray]], t: int, hw: tuple[int,int]) -> np.ndarray`: a `(t, H, W)` uint16 volume where, for branch label `L`, frame `fi`, `branch_masks[L][fi]` (bool, shape `hw`) is written as `L`. Later labels win on overlap (deterministic by ascending label).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_neuron_view.py
from __future__ import annotations
import pathlib, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import gui_neuron


def test_neurons_on_disk_groups_chains(tmp_path=None):
    d = pathlib.Path(tempfile.mkdtemp())
    for neuron, idx in [("AVAL", 0), ("AVAL", 2), ("AVAR", 1)]:
        (d / neuron / f"chain_{idx:02d}").mkdir(parents=True)
    assert gui_neuron.neurons_on_disk(d) == [("AVAL", [0, 2]), ("AVAR", [1])]


def test_build_neuron_label_volume_writes_each_branch():
    hw = (4, 4)
    b1 = np.zeros(hw, bool); b1[0:2, 0:2] = True
    b2 = np.zeros(hw, bool); b2[2:4, 2:4] = True
    vol = gui_neuron.build_neuron_label_volume({1: {0: b1}, 2: {0: b2}}, t=1, hw=hw)
    assert vol.shape == (1, 4, 4)
    assert vol[0, 0, 0] == 1 and vol[0, 3, 3] == 2 and vol[0, 0, 3] == 0


def test_build_neuron_label_volume_higher_label_wins_overlap():
    hw = (2, 2)
    a = np.ones(hw, bool)
    vol = gui_neuron.build_neuron_label_volume({1: {0: a}, 2: {0: a}}, t=1, hw=hw)
    assert (vol[0] == 2).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3 -m pytest tests/test_neuron_view.py -q`
Expected: FAIL (`No module named 'gui_neuron'`).

- [ ] **Step 3: Create `gui_neuron.py` with the module header + the two pure helpers**

```python
"""gui_neuron.py: napari NEURON-level review GUI (the second review paradigm).

The per-chain tool (gui.py) opens one chain at a time. This one opens a whole NEURON:
all its chains (branches) on a single per-neuron crop canvas (_ncrop), shown as one
multi-color object. Branches stay separate SAM2 objects; the neuron is a presentation +
union layer. See docs/superpowers/specs/2026-06-23-neuron-review-gui-design.md.

gui.py is untouched; this driver imports its shared pieces (ReviewContext, helpers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sam2_utils import review_queue


def neurons_on_disk(output_root) -> list[tuple[str, list[int]]]:
    """Every neuron with on-disk chains under output_root, as (neuron, [chain_idx,...]),
    sorted by neuron then chain. Built on ReviewQueue.all_chains so the openable set
    matches exactly what review.load_chain can read."""
    q = review_queue.ReviewQueue(Path(output_root))
    by_neuron: dict[str, list[int]] = {}
    for neuron, idx in q.all_chains():
        by_neuron.setdefault(neuron, []).append(idx)
    return [(n, sorted(by_neuron[n])) for n in sorted(by_neuron)]


def build_neuron_label_volume(branch_masks: dict, t: int,
                              hw: tuple[int, int]) -> np.ndarray:
    """(t, H, W) uint16 volume: for each branch label L and frame fi,
    branch_masks[L][fi] (bool, shape hw) is written as L. Ascending label order, so a
    higher label wins on overlap (deterministic). Labels are the per-branch editing
    integers; the saved neuron identity is independent of them."""
    H, W = hw
    vol = np.zeros((t, H, W), dtype=np.uint16)
    for label in sorted(branch_masks):
        for fi, m in branch_masks[label].items():
            if 0 <= fi < t:
                vol[fi][np.asarray(m, bool)] = label
    return vol
```

- [ ] **Step 4: Run to verify they pass**

Run: `py -3 -m pytest tests/test_neuron_view.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add gui_neuron.py tests/test_neuron_view.py
git commit -m "gui_neuron: neuron enumeration + multi-label volume builder (pure)"
```

---

## Task 4: `NeuronReviewGUI.open_neuron` (build the `_ncrop` canvas)

**Files:**
- Modify: `gui_neuron.py`

**Interfaces:**
- Consumes: `gui.ReviewContext`, `pipeline.neuron_crop_window`, `pipeline.prepare_chain_crop_frames`, `pipeline.remap_mask_to_window`, `pipeline.load_frame_sam`, `pipeline.load_state`, `sam2_utils.alignment.CropWindow`, `sam2_utils.review.load_chain`, `gui_neuron.build_neuron_label_volume`, `gui._load_frame_stack`.
- Produces: `NeuronReviewGUI(ctx, *, reviewer="")` with `open_neuron(neuron: str) -> None`.

This task has no automated test (napari + GPU frame prep). It ends with a manual launch verification. Write the code exactly as below.

- [ ] **Step 1: Add the class and `open_neuron`**

Append to `gui_neuron.py`:

```python
class NeuronReviewGUI:
    """A napari window that opens one NEURON at a time onto a per-neuron crop canvas.

    Layers per neuron:
        Image  'EM'      the _ncrop frames over the neuron's union z-range
        Labels 'neuron'  one integer per branch (selected_label = active branch)
        Points 'prompts' the active branch's click prompts
        Shapes 'box'     the active branch's bounding box
    """

    def __init__(self, ctx, *, reviewer: str = "", viewer=None):
        import napari
        from gui import ReviewContext  # noqa: F401  (type only; ctx is built by launch)
        self.ctx = ctx
        self.reviewer = reviewer
        self.viewer = viewer if viewer is not None else napari.Viewer(title="SAM2 neuron review")
        self.queue = review_queue.ReviewQueue(ctx.output_root)
        self.neuron: Optional[str] = None
        self.cw: Optional[object] = None          # the neuron CropWindow (_ncrop)
        self.chain_idxs: list[int] = []           # branch chain indices, label = idx+1
        self.frame_to_z: dict = {}
        self.frames_dir: Optional[str] = None
        self._img = self._neuron = None
        self._build_widgets()

    def open_neuron(self, neuron: str) -> None:
        import pipeline
        from sam2_utils import review
        self.neuron = neuron
        chains = [c for c in self.ctx.chains if c.get("cell_name") == neuron]
        present = [i for (n, i) in self.queue.all_chains() if n == neuron]
        self.chain_idxs = sorted(present)
        if not self.chain_idxs:
            print(f"[gui_neuron] no on-disk chains for {neuron}")
            return

        # 1. union z-range over all branches (their saved frame_to_z), and a full-frame size
        reviews = {i: review.load_chain(self.ctx.output_root / neuron / f"chain_{i:02d}")
                   for i in self.chain_idxs}
        all_z = sorted({z for rd in reviews.values() for z in rd.frame_to_z.values()})
        z_to_frame = {z: fi for fi, z in enumerate(all_z)}
        self.frame_to_z = {fi: z for z, fi in z_to_frame.items()}
        anchor_z = all_z[0]
        _img_full, full_hw = pipeline.load_frame_sam(int(anchor_z), scale=1)

        # 2. the per-neuron crop window (_ncrop)
        self.cw = pipeline.neuron_crop_window(chains, self.ctx.annotate_df,
                                              cfg=self.ctx.cfg, image_hw_tif=full_hw)
        print(f"[gui_neuron] {neuron}: {len(self.chain_idxs)} branches, {len(all_z)} slices, "
              f"_ncrop {self.cw.size_tif[0]}x{self.cw.size_tif[1]}px @ crop_scale {self.cw.crop_scale}")

        # 3. _ncrop frames over the union z-range. Reuse the chain-crop frame writer with a
        #    synthetic single-"chain" covering all the neuron's nodes (it only reads node z
        #    extent + the window slice). View dir namespaced by neuron.
        merged = {"cell_name": neuron, "nodes": [n for c in chains for n in c["nodes"]]}
        self.frames_dir, frame_to_z2, _af, _n = pipeline.prepare_chain_crop_frames(
            merged, self.ctx.annotate_df, self.cw, frames_root=self.ctx.cfg.frames_root,
            anchor_catmaid_z=int(anchor_z), neuron=neuron, chain_idx=999)
        # prepare_chain_crop_frames indexes by its own subset; rebuild our maps from it
        self.frame_to_z = frame_to_z2
        z_to_frame = {z: fi for fi, z in frame_to_z2.items()}

        # 4. remap each branch's saved masks into _ncrop, keyed by label = chain_idx + 1
        H, W = self.cw.crop_hw
        t = len(frame_to_z2)
        branch_masks: dict[int, dict[int, np.ndarray]] = {}
        for i in self.chain_idxs:
            rd = reviews[i]
            sp = self.ctx.output_root / neuron / f"chain_{i:02d}" / "state.json"
            st = pipeline.load_state(sp) if sp.exists() else None
            from sam2_utils.alignment import CropWindow
            src_cw = (CropWindow.from_dict(st.crop_window)
                      if st is not None and getattr(st, "crop_window", None) else None)
            label = i + 1
            branch_masks[label] = {}
            for fi_src, z in rd.frame_to_z.items():
                fi = z_to_frame.get(z)
                if fi is None or fi_src not in rd.video_segments:
                    continue
                seg = rd.video_segments[fi_src].get(rd.obj_id)
                if seg is None:
                    continue
                m = np.asarray(seg)
                m = m[0] if m.ndim == 3 else m
                if src_cw is not None:           # tier-2 branch: _pcrop footprint
                    so, ss = src_cw.origin_tif, src_cw.size_tif
                else:                            # legacy _sam branch: full-frame footprint
                    so = (0.0, 0.0)
                    ss = (full_hw[1], full_hw[0])
                branch_masks[label][fi] = pipeline.remap_mask_to_window(
                    m, src_origin_tif=so, src_size_tif=ss, dst_cw=self.cw)

        vol = build_neuron_label_volume(branch_masks, t, (H, W))

        # 5. (re)build layers
        em = self._load_ncrop_stack(t)
        self.viewer.layers.clear()
        self._img = self.viewer.add_image(em, name="EM", rgb=True)
        self._neuron = self.viewer.add_labels(vol, name="neuron", opacity=0.5)
        self._neuron.selected_label = (self.chain_idxs[0] + 1)
        self.viewer.reset_view()
        self._refresh_info()

    def _load_ncrop_stack(self, t: int):
        from gui import _load_frame_stack
        return _load_frame_stack(self.frames_dir, t)
```

- [ ] **Step 2: Add a minimal `_build_widgets` and `_refresh_info` so the window opens**

```python
    def _build_widgets(self) -> None:
        from magicgui.widgets import Container, ComboBox, PushButton, Label
        self._info = Label(value="(no neuron open)")
        neurons = [n for (n, _idxs) in neurons_on_disk(self.ctx.output_root)] or ["(none)"]
        self._neuron_combo = ComboBox(label="neuron", choices=neurons)
        open_btn = PushButton(text="open neuron")
        open_btn.changed.connect(lambda *_: self.open_neuron(str(self._neuron_combo.value)))
        panel = Container(widgets=[self._neuron_combo, open_btn, self._info], labels=True)
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel.native)
        self.viewer.window.add_dock_widget(scroll, area="right", name="neuron review")

    def _refresh_info(self) -> None:
        if self.neuron is None:
            self._info.value = "(no neuron open)"
            return
        self._info.value = (f"{self.neuron}\n{len(self.chain_idxs)} branches\n"
                            f"active branch (label) = {getattr(self._neuron, 'selected_label', '?')}")
```

- [ ] **Step 3: Add a `launch` entry point + `main`**

```python
def launch(output_root: Optional[Path] = None, *, neuron: Optional[str] = None,
           reviewer: str = "", block: bool = True):
    """Open the neuron-review GUI. With ``neuron`` it opens straight onto one neuron."""
    import napari
    from gui import ReviewContext
    from sam2_utils import config
    ctx = ReviewContext(Path(output_root) if output_root else config.OUTPUT_ROOT)
    gui = NeuronReviewGUI(ctx, reviewer=reviewer)
    if neuron is not None:
        gui.open_neuron(neuron)
    if block:
        napari.run()
    return gui


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="SAM2 napari NEURON-level review GUI")
    ap.add_argument("--output-root", type=str, default=None)
    ap.add_argument("--neuron", type=str, default=None)
    ap.add_argument("--reviewer", type=str, default="")
    args = ap.parse_args()
    launch(Path(args.output_root) if args.output_root else None,
           neuron=args.neuron, reviewer=args.reviewer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lint + import check**

Run: `py -3 -m ruff check gui_neuron.py && py -3 -c "import gui_neuron; print('ok', hasattr(gui_neuron.NeuronReviewGUI, 'open_neuron'))"`
Expected: `All checks passed!` then `ok True`.

- [ ] **Step 5: Manual launch verification (needs GPU-free; frame prep only)**

Run: `py -3 gui_neuron.py --neuron AVAL`
Expected: a napari window opens; the console prints `[gui_neuron] AVAL: N branches, M slices, _ncrop WxH @ crop_scale S`; the EM shows the neuron crop and the `neuron` Labels layer shows each branch in a distinct color. Scrub z to confirm branches appear on their slices. Close the window.

- [ ] **Step 6: Commit**

```bash
git add gui_neuron.py
git commit -m "gui_neuron: open a neuron onto the _ncrop canvas with per-branch labels"
```

---

## Task 5: Active-branch correction (prompts/box + re-predict/resume in `_ncrop`)

**Files:**
- Modify: `gui_neuron.py`

**Interfaces:**
- Consumes: `pipeline.PropagationSession`, `pipeline.image_predict`, `pipeline.Prompts`, `pipeline.save_masks`, `pipeline.run_qc`, the prompt/box helpers and `_prompts_for_frame`/`_box_for_frame` logic from `gui.py` (copied or imported), `gui_neuron.NeuronReviewGUI` from Task 4.
- Produces: `rerun_image_phase`, `resume_propagation`, `_active_label`, `_set_branch_mask`, `_save_branch` methods; a prompts Points layer and a box Shapes layer scoped to `selected_label`; `b`/`r`/`g` key bindings.

This task is napari + GPU; it ends with a manual verification, no CI test. The active branch is `self._neuron.selected_label`; its `obj_id` is the branch's saved `obj_id` (read from `review.load_chain`), and its frames are the shared `_ncrop` frames. Build one `PropagationSession` over `self.frames_dir` per active branch lazily, keyed by label, closing the previous on switch.

- [ ] **Step 1: Add prompt/box layers (in `open_neuron`, after the EM/neuron layers) + active-branch helpers**

Add to `open_neuron` after `self._neuron.selected_label = ...`:

```python
        import pandas as pd
        from gui import _POS_COLOR, _NEG_COLOR, _PROMPT_LABELS, _BOX_EDGE_COLOR
        feats = pd.DataFrame({"label": pd.Categorical([], categories=_PROMPT_LABELS)})
        self._prompts = self.viewer.add_points(
            np.empty((0, 3)), name="prompts", ndim=3, size=6.0, features=feats,
            border_color="label", border_color_cycle=[_POS_COLOR, _NEG_COLOR],
            face_color="transparent", border_width=0.4, symbol="o")
        self._prompts.border_color_mode = "cycle"
        self._prompts.feature_defaults = {"label": "positive"}
        self._prompts.mode = "add"
        self._box = self.viewer.add_shapes(
            name="box", ndim=3, edge_color=_BOX_EDGE_COLOR, face_color="transparent",
            edge_width=1.0, opacity=0.8)
```

And these methods (the prompt/box readers are the per-chain GUI's logic with `scale=(1,1,1)` since `_ncrop` is the data grid):

```python
    def _active_label(self) -> int:
        return int(getattr(self._neuron, "selected_label", 0) or 0)

    def _active_obj_id(self) -> int:
        from sam2_utils import review
        idx = self._active_label() - 1
        rd = review.load_chain(self.ctx.output_root / self.neuron / f"chain_{idx:02d}")
        return int(rd.obj_id)

    def _prompts_for_frame(self, fi: int):
        import pipeline
        from gui import _LABEL_TO_SAM
        data = np.asarray(self._prompts.data, dtype=float)
        if not len(data):
            return pipeline.Prompts(points_sam=np.empty((0, 2)), labels=np.empty((0,), int))
        on = np.round(data[:, 0]).astype(int) == int(fi)
        rows = data[on]
        feats = self._prompts.features.get("label", None)
        labs_src = np.asarray(feats)[on] if feats is not None else ["positive"] * len(rows)
        pts_xy = np.stack([rows[:, 2], rows[:, 1]], axis=1)
        labs = np.array([_LABEL_TO_SAM.get(str(s), 1) for s in labs_src], dtype=int)
        return pipeline.Prompts(points_sam=pts_xy, labels=labs)

    def _box_for_frame(self, fi: int):
        from gui import _box_on_frame
        return None if self._box is None else _box_on_frame(self._box.data, int(fi))
```

Also add to `__init__` (after `_build_widgets`): `self._prompts = self._box = None; self._active_session = None; self._session_label = None; self._bind_keys()`.

- [ ] **Step 2: Add `rerun_image_phase` (re-predict the active branch in `_ncrop`)**

```python
    def rerun_image_phase(self, *_):
        import pipeline
        from sam2_utils.video_viz import _load_frame
        if self.neuron is None:
            return
        fi = int(self.viewer.dims.current_step[0])
        prompts = self._prompts_for_frame(fi)             # in _ncrop coords (see gui.py)
        prompts.box_sam = self._box_for_frame(fi)
        if not (prompts.labels == 1).any() and prompts.box_sam is None:
            print("[gui_neuron] need a positive point or a box on this frame")
            return
        self.ctx.ensure_predictors(need_image=True, need_video=False)
        em = _load_frame(self.frames_dir, fi)             # the _ncrop frame
        mask, score, _ = pipeline.image_predict(self.ctx.image_predictor, em, prompts)
        self.ctx.image_predictor.reset_predictor()
        self._set_branch_mask(fi, mask)
        print(f"[gui_neuron] re-predicted branch {self._active_label()} frame {fi}: "
              f"{int(mask.sum())} px, score {score:.3f}")

    def _set_branch_mask(self, fi: int, mask):
        lbl = self._active_label()
        vol = self._neuron.data
        m = np.asarray(mask, bool)
        frame = vol[fi]
        frame[frame == lbl] = 0          # clear this branch's old pixels on this frame
        frame[m] = lbl
        self._neuron.data = vol
        self._neuron.refresh()
```

- [ ] **Step 3: Add `resume_propagation` (re-track the active branch over `_ncrop` frames)**

```python
    def _session(self):
        import pipeline
        lbl = self._active_label()
        if getattr(self, "_session_label", None) != lbl:
            old = getattr(self, "_active_session", None)
            if old is not None:
                old.close()
            self._active_session = pipeline.PropagationSession(
                self.ctx.video_predictor, self.frames_dir, obj_id=self._active_obj_id())
            self._session_label = lbl
        return self._active_session

    def resume_propagation(self, *_):
        if self.neuron is None:
            return
        self.ctx.ensure_predictors(need_image=False, need_video=True)
        fi = int(self.viewer.dims.current_step[0])
        lbl = self._active_label()
        mask = (self._neuron.data[fi] == lbl)
        if not mask.any():
            print("[gui_neuron] no mask for the active branch on this frame; re-predict first")
            return
        sess = self._session()
        sess.add_mask(fi, mask)
        for rev in (False, True):
            for _ in sess.propagate(reverse=rev, start_frame_idx=fi):
                pass
        for f2, seg in sess.video_segments.items():
            if sess.obj_id in seg:
                mm = np.asarray(seg[sess.obj_id]); mm = mm[0] if mm.ndim == 3 else mm
                self._set_branch_mask(f2, mm.astype(bool))
        self._save_branch(lbl)
        print(f"[gui_neuron] branch {lbl} re-propagated and saved")
```

- [ ] **Step 4: Add `_save_branch` (write the branch's `_ncrop` masks + state)**

```python
    def _save_branch(self, label: int):
        import pipeline
        idx = label - 1
        chain_dir = self.ctx.output_root / self.neuron / f"chain_{idx:02d}"
        obj_id = self._active_obj_id()
        segments = {fi: {obj_id: (self._neuron.data[fi] == label)}
                    for fi in range(self._neuron.data.shape[0])
                    if (self._neuron.data[fi] == label).any()}
        pipeline.save_masks(segments, self.frame_to_z, chain_dir / "masks",
                            obj_id=obj_id, mask_space_downscale=self.ctx.cfg.save_downscale)
        sp = chain_dir / "state.json"
        if sp.exists():
            st = pipeline.load_state(sp)
            st.crop_window = self.cw.to_dict()      # branch now lives in _ncrop
            pipeline.save_state(st, sp)
        self.queue.set_status(self.neuron, idx, review_queue.CORRECTED, reviewer=self.reviewer)
```

- [ ] **Step 5: Bind keys (`b`/`r`/`g`) and lint/import**

```python
    def _bind_keys(self):
        v = self.viewer
        @v.bind_key("r", overwrite=True)
        def _r(_v): self.rerun_image_phase()
        @v.bind_key("g", overwrite=True)
        def _g(_v): self.resume_propagation()
        @v.bind_key("b", overwrite=True)
        def _b(_v):
            if self._box is not None:
                self.viewer.layers.selection.active = self._box
                self._box.mode = "add_rectangle"
```

Run: `py -3 -m ruff check gui_neuron.py && py -3 -c "import gui_neuron"`
Expected: clean, imports.

- [ ] **Step 6: Manual verification (GPU)**

Run: `py -3 gui_neuron.py --neuron AVAL`
Pick the active branch via the `neuron` layer's label control, drop a positive point on a frame, press `R` (mask updates for that branch only), then `G` (branch re-propagates and saves). Confirm other branches are untouched and `_review.csv` shows the branch `corrected`.

- [ ] **Step 7: Commit**

```bash
git add gui_neuron.py
git commit -m "gui_neuron: active-branch re-predict + resume in the _ncrop canvas"
```

---

## Task 6: Neuron-level disposition + docs + code-map

**Files:**
- Modify: `gui_neuron.py` (add approve/reject across the neuron's branches)
- Create: `docs/how-to/review-a-neuron.md`
- Modify: `docs/reference/code-map.md`

**Interfaces:**
- Consumes: `review_queue.ReviewQueue.set_status`, `APPROVED`, `REJECTED`.
- Produces: `approve_neuron`, `reject_neuron` (set every branch's review status), and dock buttons.

- [ ] **Step 1: Add neuron-level disposition**

```python
    def approve_neuron(self, *_):
        for i in self.chain_idxs:
            self.queue.set_status(self.neuron, i, review_queue.APPROVED, reviewer=self.reviewer)
        print(f"[gui_neuron] {self.neuron} approved ({len(self.chain_idxs)} branches)")

    def reject_neuron(self, *_):
        for i in self.chain_idxs:
            self.queue.set_status(self.neuron, i, review_queue.REJECTED, reviewer=self.reviewer)
        print(f"[gui_neuron] {self.neuron} rejected ({len(self.chain_idxs)} branches)")
```

Add `approve neuron` / `reject neuron` PushButtons to `_build_widgets`'s panel, wired to these.

- [ ] **Step 2: Write the how-to**

Create `docs/how-to/review-a-neuron.md` documenting: launch (`py -3 gui_neuron.py --neuron AVAL`), the single multi-color `neuron` layer (label = branch, selected label = active branch), `R`/`G`/`B` act on the active branch in the `_ncrop` canvas, neuron-level approve/reject, the resolution note (a large neuron gets a coarser canvas; use `gui.py` for full-sharpness single-branch work), and that opening a neuron preps full-res frames so it is not instant. Run the `humanizer` skill on it; no em dashes.

- [ ] **Step 3: Update the code-map**

In `docs/reference/code-map.md`, add a row: `gui_neuron.py` -> "neuron-level review GUI (second paradigm); per-neuron _ncrop canvas, branches as labels; see the 2026-06-23 spec".

- [ ] **Step 4: Full verify**

Run: `py -3 -m pytest -q && py -3 -m ruff check .`
Expected: all pass, lint clean. Confirm no em dashes in touched files.

- [ ] **Step 5: Commit**

```bash
git add gui_neuron.py docs/how-to/review-a-neuron.md docs/reference/code-map.md
git commit -m "gui_neuron: neuron-level disposition + docs"
```

---

## Notes for the implementer

- **Spaces:** the `_ncrop` canvas IS the data grid for every layer, so prompts/box/paint coordinates are `_ncrop` directly (no transform), exactly as the per-chain GUI treats its displayed frame. `image_predict` and the session run on the `_ncrop` frames, so they consume `_ncrop` coords too.
- **The `chain_idx=999` synthetic view dir** in Task 4 just namespaces the neuron's `_ncrop` frame cache; it does not collide with real chains. If you prefer, pass `chain_idx=-1` or a neuron-specific tag; keep it stable so reopening reuses the cache.
- **`prepare_chain_crop_frames` returns its own `frame_to_z`** over the node z-extent; Task 4 rebuilds the maps from it (step 3). Do not assume it matches a single branch's z-range.
- **Resolution trade is expected:** a large neuron yields a coarse `_ncrop`. That is by design (spec, Decided scope #3). Direct the reviewer to `gui.py` for a single thin branch needing full tier-2 sharpness.
- **Deferred (not in this plan):** cross-neuron overlap arbitration, and any change to `gui.py` or the batch.
