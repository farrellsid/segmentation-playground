# Per-frame segmentation, Plan 1: foundation + Approach 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared per-frame foundation (node index, scoring, two overlap resolvers, a metric-guided candidate selector) and Approach 1 (prompt-based per-frame segmentation), plus the eval/montage harness, as a complete working per-frame pipeline.

**Architecture:** Pure primitives (torch-free) live in `sam2_utils/perframe.py` and `eval/perframe_score.py`; the SAM2-touching runner lives in a new `run_perframe.py` driver. Approach 1 prompts SAM2 image-mode once per node in a frame, collects labelled masks, resolves overlaps membrane-aware, and scores against our metric. Design: `docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md`.

**Tech Stack:** Python, numpy, scipy.ndimage, scikit-image, SAM2 image predictor, pytest.

## Global Constraints

- No em dashes anywhere (code, comments, docstrings, commit messages).
- Tests CPU-only and torch-free: `py -3 -m pytest`. The runner gets a `--model-size tiny` CPU smoke; real runs are GPU/CCDB.
- Lint with `ruff check .`; only clean files you touch.
- The library (`pipeline/`, `sam2_utils/`) must never import the drivers (`batch`, `gui`, `run_aval`, `run_perframe`) or `eval`. `tests/test_import_direction.py` enforces this. `run_perframe.py` is a driver.
- Downscale for local iteration (scale 8). Every experiment run keeps `results/perframe/<run>/` (config.json, scores.csv, montages/) and appends a line to `docs/explanation/perframe-experiments.md`.
- F2 composite objective (shared by scorer, selector, tuner): require own-node containment and zero foreign nodes, then maximise `boundary_on_membrane`, then minimise `spanning_merge` and overlap.

---

### Task 1: F1 per-frame node index

**Files:**
- Create: `sam2_utils/perframe.py`
- Test: `tests/test_perframe.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `nodes_in_frame(annotate_df, catmaid_z, scale) -> list[tuple[float, float, str, str]]` returning `(x_sam, y_sam, cell_name, node_id)` for every node at that z, coords divided by scale (same grid as `merge_metric.nodes_by_z`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perframe.py
import numpy as np
import pandas as pd
from sam2_utils import perframe as pf


def test_nodes_in_frame_filters_z_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"], "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401], "x_tif": [800.0, 1600.0, 240.0], "y_tif": [80.0, 160.0, 800.0],
    })
    got = pf.nodes_in_frame(df, 1400, scale=8)
    assert sorted(got) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert pf.nodes_in_frame(df, 1401, scale=8) == [(30.0, 100.0, "AVAL", "c")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_perframe.py::test_nodes_in_frame_filters_z_and_scales -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sam2_utils.perframe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# sam2_utils/perframe.py
"""Per-frame segmentation primitives (torch-free): node index, overlap resolution,
metric-guided candidate selection, and AMG-to-node matching. The SAM2-touching runner
lives in run_perframe.py. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def nodes_in_frame(annotate_df, catmaid_z: int, scale: int
                   ) -> list[tuple[float, float, str, str]]:
    """Every node at catmaid_z across all neurons, as (x_sam, y_sam, cell_name, node_id).
    Coords are x_tif/scale (the _sam grid merge_metric uses)."""
    z = annotate_df["z"].astype(int)
    sub = annotate_df[z == int(catmaid_z)]
    out = []
    for _, r in sub.iterrows():
        out.append((float(r["x_tif"]) / scale, float(r["y_tif"]) / scale,
                    str(r["cell_name"]), str(r["node_id"])))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_perframe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/perframe.py tests/test_perframe.py
git commit -m "feat(perframe): per-frame node index (F1)"
```

---

### Task 2: F3 overlap resolvers (argmax + watershed)

**Files:**
- Modify: `sam2_utils/perframe.py`
- Test: `tests/test_perframe.py`

**Interfaces:**
- Consumes: F1 module.
- Produces: `resolve_overlaps_argmax(masks, node_xy, membrane_map=None) -> np.ndarray` (int label map, 0 = background, i+1 = masks[i]) and `resolve_overlaps_watershed(masks, node_xy, membrane_map) -> np.ndarray`, both taking `masks: list[np.ndarray bool]` and `node_xy: list[(x, y)]` (one seed per mask, same order).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_perframe.py

def test_argmax_resolves_overlap_to_nearest_node():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = np.zeros((20, 20), bool); b[8:18, 8:18] = True   # overlaps a in [8:12, 8:12]
    lab = pf.resolve_overlaps_argmax([a, b], [(6, 6), (13, 13)])
    # contested pixel (9,9): nearer node b(13,13)? dist to (6,6)=~4.2, to (13,13)=~5.6 -> a
    assert lab[9, 9] == 1
    # (11,11): to (6,6)=~7.1, to (13,13)=~2.8 -> b
    assert lab[11, 11] == 2
    # uncontested
    assert lab[3, 3] == 1 and lab[16, 16] == 2 and lab[0, 0] == 0


def test_watershed_labels_are_disjoint_and_seeded():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = np.zeros((20, 20), bool); b[8:18, 8:18] = True
    mem = np.zeros((20, 20), np.float32)
    lab = pf.resolve_overlaps_watershed([a, b], [(6, 6), (13, 13)], mem)
    assert lab[6, 6] == 1 and lab[13, 13] == 2          # seeds keep their label
    assert set(np.unique(lab)) <= {0, 1, 2}
    assert not ((lab == 1) & (lab == 2)).any()          # disjoint by construction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_perframe.py -k "argmax or watershed" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sam2_utils/perframe.py

def resolve_overlaps_argmax(masks, node_xy, membrane_map=None) -> np.ndarray:
    """Assign one label per pixel. Uncontested pixels keep their only claimant. A pixel
    claimed by several masks goes to the claimant whose seed node is nearest (Euclidean).
    membrane_map is accepted for signature parity with the watershed resolver and is not
    used by the nearest-node rule. Returns an int label map (0 background, i+1 = masks[i])."""
    if not masks:
        raise ValueError("no masks")
    h, w = masks[0].shape
    stack = np.stack([m.astype(bool) for m in masks], axis=0)   # (K, H, W)
    count = stack.sum(axis=0)
    lab = np.zeros((h, w), dtype=np.int32)
    # uncontested: exactly one claimant -> that label (argmax of the one True plane)
    single = count == 1
    lab[single] = stack[:, single].argmax(axis=0) + 1
    # contested: nearest seed node among claimants
    ys, xs = np.where(count > 1)
    if ys.size:
        seeds = np.asarray(node_xy, dtype=float)                # (K, 2) as (x, y)
        for y, x in zip(ys, xs):
            claim = np.where(stack[:, y, x])[0]
            d = (seeds[claim, 0] - x) ** 2 + (seeds[claim, 1] - y) ** 2
            lab[y, x] = int(claim[int(np.argmin(d))]) + 1
    return lab


def resolve_overlaps_watershed(masks, node_xy, membrane_map) -> np.ndarray:
    """Seeded watershed on the membrane map: each seed node is a marker, the membrane map
    is the elevation (walls at ridges), flooding restricted to the union of the masks.
    Returns an int label map (0 background, i+1 = masks[i])."""
    from skimage.segmentation import watershed
    if not masks:
        raise ValueError("no masks")
    h, w = masks[0].shape
    union = np.zeros((h, w), bool)
    for m in masks:
        union |= m.astype(bool)
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (x, y) in enumerate(node_xy):
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < h and 0 <= xi < w:
            markers[yi, xi] = i + 1
    elevation = (membrane_map if membrane_map is not None
                 else np.zeros((h, w), np.float32)).astype(np.float32)
    lab = watershed(elevation, markers=markers, mask=union)
    return lab.astype(np.int32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_perframe.py -k "argmax or watershed" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/perframe.py tests/test_perframe.py
git commit -m "feat(perframe): argmax + watershed overlap resolvers (F3)"
```

---

### Task 3: Metric-guided candidate selector + AMG-to-node matcher

**Files:**
- Modify: `sam2_utils/perframe.py`
- Test: `tests/test_perframe.py`

**Interfaces:**
- Consumes: `sam2_utils.membrane` (`boundary_on_membrane`, `spanning_membrane`); `pipeline._point_in_mask`.
- Produces: `select_by_metric(candidates, node_xy, foreign_xy, membrane_map, *, radius=3, tau=0.5) -> int` (index of the best candidate, or -1 if none qualify); `match_amg_to_nodes(amg_masks, node_index, membrane_map, *, radius=3, tau=0.5) -> (labels: dict[str, np.ndarray], leftover: list[np.ndarray])`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_perframe.py

def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_select_by_metric_prefers_node_containing_membrane_aligned():
    node = (20, 20)
    small = _disk(20, 20, 4)          # contains node, tight
    big = _disk(20, 20, 15)           # contains node, but engulfs a foreign node
    off = _disk(35, 35, 4)            # does not contain node
    mem = np.zeros((40, 40), np.float32)
    # membrane ridge on the small disk's rim -> high boundary_on_membrane for `small`
    from sam2_utils.perframe import _rim
    mem[_rim(small)] = 1.0
    idx = pf.select_by_metric([off, big, small], node, foreign_xy=[(20, 30)], membrane_map=mem)
    assert idx == 2                   # `small`: contains node, no foreign, best boundary


def test_match_amg_assigns_nodes_and_keeps_leftover():
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    m_a = _disk(10, 10, 5); m_b = _disk(30, 30, 5); junk = _disk(20, 5, 3)
    mem = np.zeros((40, 40), np.float32)
    labels, leftover = pf.match_amg_to_nodes([junk, m_a, m_b], node_index, mem)
    assert set(labels) == {"AVAL", "AVAR"}
    assert int(labels["AVAL"].sum()) == int(m_a.sum())
    assert len(leftover) == 1 and int(leftover[0].sum()) == int(junk.sum())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_perframe.py -k "select_by_metric or match_amg" -v`
Expected: FAIL with `AttributeError` (and `_rim` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# append to sam2_utils/perframe.py
from sam2_utils import membrane as _mb


def _rim(mask: np.ndarray) -> np.ndarray:
    """1-px inner boundary of a bool mask (shared by tests + boundary scoring)."""
    return mask & ~ndi.binary_erosion(mask)


def _contains(mask: np.ndarray, xy, radius: int) -> bool:
    import pipeline
    return pipeline._point_in_mask(mask, float(xy[0]), float(xy[1]), radius)


def select_by_metric(candidates, node_xy, foreign_xy, membrane_map, *,
                     radius: int = 3, tau: float = 0.5) -> int:
    """Index of the candidate that best satisfies the F2 composite: must contain node_xy
    and no foreign node; among those, maximise boundary_on_membrane, then minimise the
    spanning-membrane bled_fraction. Returns -1 if none contain the node without a foreign
    hit."""
    best, best_key = -1, None
    for i, m in enumerate(candidates):
        m = m.astype(bool)
        if not _contains(m, node_xy, radius):
            continue
        if any(_contains(m, f, radius) for f in foreign_xy):
            continue
        bo = _mb.boundary_on_membrane(m, membrane_map, tau=tau)
        _, bled = _mb.spanning_membrane(m, membrane_map, tau=tau)
        key = (bo, -bled)                       # higher boundary, lower bled
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def match_amg_to_nodes(amg_masks, node_index, membrane_map, *,
                       radius: int = 3, tau: float = 0.5):
    """Assign each node its AMG mask (the containing mask best on the F2 composite, via
    select_by_metric with the OTHER nodes as foreign), label it by cell_name, and return
    the leftover (unmatched) masks as unlabelled competitors."""
    labels: dict[str, np.ndarray] = {}
    used = set()
    for (x, y, cell, _nid) in node_index:
        foreign = [(fx, fy) for (fx, fy, fc, _f) in node_index if fc != cell]
        idx = select_by_metric(amg_masks, (x, y), foreign, membrane_map,
                               radius=radius, tau=tau)
        if idx >= 0:
            labels[cell] = amg_masks[idx].astype(bool)
            used.add(idx)
    leftover = [m.astype(bool) for i, m in enumerate(amg_masks) if i not in used]
    return labels, leftover
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_perframe.py -k "select_by_metric or match_amg" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/perframe.py tests/test_perframe.py
git commit -m "feat(perframe): metric-guided candidate selector + AMG-to-node matcher"
```

---

### Task 4: F2 per-frame scoring

**Files:**
- Create: `eval/perframe_score.py`
- Test: `tests/test_perframe_score.py`

**Interfaces:**
- Consumes: `sam2_utils.membrane`, `pipeline._point_in_mask`, F1 `nodes_in_frame`.
- Produces: `score_frame(cell_masks: dict[str, np.ndarray], node_index, membrane_map=None, *, radius=3, tau=0.5) -> dict` with keys `own_coverage`, `foreign_frame_rate`, `total_foreign`, `mean_boundary_on_membrane`, `spanning_rate`, `mean_underfill`, `overlap_fraction`, and `per_cell` (list of dicts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perframe_score.py
import numpy as np
from eval.perframe_score import score_frame


def _disk(cx, cy, r, shape=(60, 60)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_score_frame_own_foreign_and_overlap():
    node_index = [(15, 15, "AVAL", "a"), (40, 40, "AVAR", "b")]
    masks = {"AVAL": _disk(15, 15, 8), "AVAR": _disk(40, 40, 8)}   # disjoint, each own node
    s = score_frame(masks, node_index, membrane_map=None)
    assert s["own_coverage"] == 1.0
    assert s["total_foreign"] == 0
    assert s["overlap_fraction"] == 0.0
    # now make AVAL swallow AVAR's node -> a foreign hit and overlap
    masks2 = {"AVAL": _disk(27, 27, 22), "AVAR": _disk(40, 40, 8)}
    s2 = score_frame(masks2, node_index, membrane_map=None)
    assert s2["total_foreign"] >= 1
    assert s2["overlap_fraction"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_perframe_score.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/perframe_score.py
"""Per-frame scoring metric (F2): grade a frame's labelled instance masks with the same
GT-free primitives merge_metric + membrane use, plus a pre-resolution overlap scalar.
Torch-free; doubles as the tuner objective and the metric-guided selector's basis."""
from __future__ import annotations

import numpy as np

import pipeline
from sam2_utils import membrane as mb


def _contains(mask, xy, radius):
    return pipeline._point_in_mask(mask, float(xy[0]), float(xy[1]), radius)


def score_frame(cell_masks, node_index, membrane_map=None, *, radius=3, tau=0.5) -> dict:
    """cell_masks: {cell_name: bool mask}. node_index: [(x, y, cell, node_id), ...] for this
    frame (F1). membrane_map: float [0,1] frame map or None (membrane columns then NaN)."""
    cells = list(cell_masks)
    per = []
    for cell in cells:
        m = cell_masks[cell].astype(bool)
        own = [(x, y) for (x, y, c, _n) in node_index if c == cell]
        foreign = [(x, y) for (x, y, c, _n) in node_index if c != cell]
        own_ok = any(_contains(m, xy, radius) for xy in own) if own else False
        n_foreign = sum(_contains(m, xy, radius) for xy in foreign)
        row = {"cell": cell, "own_contained": bool(own_ok), "n_foreign": int(n_foreign),
               "area": int(m.sum())}
        if membrane_map is not None and m.any():
            sp, bled = mb.spanning_membrane(m, membrane_map, tau=tau)
            row["spanning"] = bool(sp)
            row["boundary_on_membrane"] = float(mb.boundary_on_membrane(m, membrane_map, tau=tau))
            row["underfill"] = float(mb.underfill_fraction(m, membrane_map, tau=tau))
        per.append(row)

    n = len(per)
    total_area = float(sum(r["area"] for r in per)) or 1.0
    # pairwise overlap fraction (pre-resolution fight for pixels)
    overlap = 0
    ms = [cell_masks[c].astype(bool) for c in cells]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            overlap += int((ms[i] & ms[j]).sum())
    have_mem = any("boundary_on_membrane" in r for r in per)
    summary = {
        "n_cells": n,
        "own_coverage": float(np.mean([r["own_contained"] for r in per])) if n else 0.0,
        "foreign_frame_rate": float(np.mean([r["n_foreign"] > 0 for r in per])) if n else 0.0,
        "total_foreign": int(sum(r["n_foreign"] for r in per)),
        "overlap_fraction": float(overlap / total_area),
        "mean_boundary_on_membrane": (float(np.mean([r["boundary_on_membrane"] for r in per
                                                     if "boundary_on_membrane" in r]))
                                      if have_mem else None),
        "spanning_rate": (float(np.mean([r["spanning"] for r in per if "spanning" in r]))
                          if have_mem else None),
        "mean_underfill": (float(np.mean([r["underfill"] for r in per if "underfill" in r]))
                           if have_mem else None),
        "per_cell": per,
    }
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_perframe_score.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint, commit**

Run: `py -3 -m pytest -q && ruff check eval/perframe_score.py sam2_utils/perframe.py tests/test_perframe.py tests/test_perframe_score.py`

```bash
git add eval/perframe_score.py tests/test_perframe_score.py
git commit -m "feat(eval): per-frame scoring metric (F2)"
```

---

### Task 5: Approach 1 runner (`run_perframe.py`, prompt mode)

**Files:**
- Create: `run_perframe.py`
- Test: `tests/test_run_perframe_smoke.py` (CPU, tiny model, 1 downscaled frame)

**Interfaces:**
- Consumes: `sam2_utils.setup.build_predictor`, `pipeline.load_frame_sam`, `pipeline.build_prompts`, `pipeline.image_predict`, `sam2_utils.perframe` (F1, F3, selector), `sam2_utils.membrane.membrane_map`, `eval.perframe_score.score_frame`, `merge_metric.load_node_table`.
- Produces: `segment_frame_prompt(image_predictor, frame_sam, node_index, membrane_map, *, negatives, selection, resolver, cfg) -> (cell_masks, label_map, score)`; a CLI `run_perframe.py --approach prompt --frames <z...> --negatives on|off --selection pred_iou|generous|metric --resolver argmax|watershed --scale 8 --model-size <s> --out results/perframe/<run>`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_run_perframe_smoke.py
import numpy as np
import pytest
torch = pytest.importorskip("torch")   # smoke only runs where torch is present
import run_perframe


def test_segment_frame_prompt_shapes(monkeypatch):
    # a fake image predictor: returns one candidate mask per set_image/predict, torch-free
    class FakePred:
        def set_image(self, img): self._hw = img.shape[:2]
        def predict(self, **kw):
            h, w = self._hw
            m = np.zeros((1, h, w), bool); m[0, :h // 2, :w // 2] = True
            return m, np.array([0.9]), np.zeros((1, 256, 256), np.float32)
        def reset_predictor(self): pass
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    cell_masks, lab, score = run_perframe.segment_frame_prompt(
        FakePred(), frame, node_index, mem, negatives=True, selection="pred_iou",
        resolver="argmax", cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_run_perframe_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_perframe'`.

- [ ] **Step 3: Write the runner**

Create `run_perframe.py`. Structure (fill with the exact calls named in Interfaces):

```python
"""Per-frame neuron segmentation driver. Approach 1 (prompt-based) here; Approach 2 (AMG)
in a later change. Segments every node-bearing cell in a frame, resolves overlaps
membrane-aware, scores with eval.perframe_score, and writes results/montages. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sam2_utils import perframe as pf, membrane as mb
from eval.perframe_score import score_frame


@dataclass
class PerframeCfg:
    scale: int = 8
    radius: int = 3
    tau: float = 0.5
    k_max_neg: int = 3
    box_margin: int = 10


def segment_frame_prompt(image_predictor, frame_sam, node_index, membrane_map, *,
                         negatives: bool, selection: str, resolver: str, cfg: PerframeCfg):
    """One frame, prompt mode. For each node: set_image, predict with a positive point (+
    box, + the OTHER cells' nodes as negatives when `negatives`), take SAM2's 3 candidates,
    pick one by `selection` (pred_iou | generous | metric), collect labelled masks, resolve
    overlaps by `resolver` (argmax | watershed), score. Returns (cell_masks, label_map, score).

    Implementation notes:
    - set_image(frame_sam) once, reuse across nodes (same frame).
    - positive point = (x, y) of the node in _sam; negatives = other nodes' (x, y) with
      label 0 when `negatives`; box from a first-pass single-mask predict (mirror
      pipeline.box_from_mask) or omit for the smoke.
    - predict(multimask_output=True) -> (masks[3], scores[3], logits). selection:
        pred_iou   -> argmax(scores)
        generous   -> largest-area candidate that contains the node (mirror multimask_generous)
        metric     -> pf.select_by_metric(list(masks), node_xy, foreign_xy, membrane_map)
      foreign_xy = the other cells' node coords.
    - cell_masks[cell] = chosen mask; label_map = pf.resolve_overlaps_{argmax,watershed}(
      masks_in_node_order, node_xy_in_order, membrane_map).
    - score = score_frame(cell_masks, node_index, membrane_map, radius=cfg.radius, tau=cfg.tau).
    """
    # ... (per the notes above; the FakePred smoke exercises the control flow)


def _run(args):
    """Build the image predictor (sam2_utils.setup.build_predictor size, kind='image'),
    load node table (merge_metric.load_node_table), for each --frames z:
    load_frame_sam(z, scale), membrane_map(frame gray), nodes_in_frame, segment_frame_prompt,
    write results/perframe/<run>/{config.json, scores.csv, montages/<z>.png}, append a line
    to docs/explanation/perframe-experiments.md."""
    # ... argparse: --approach prompt --frames --negatives --selection --resolver --scale
    #     --model-size --out ; montage = EM | coloured label_map | membrane overlay.


if __name__ == "__main__":
    raise SystemExit(_run(_parse()))
```

The smoke test only exercises `segment_frame_prompt` with a fake predictor, so implement that function fully; `_run` / argparse / montage can be built out and are covered by the CPU smoke run in Step 5, not the unit test.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_run_perframe_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: CPU smoke on a real downscaled frame**

Run: `py -3 run_perframe.py --approach prompt --frames 1400 --negatives on --selection metric --resolver argmax --scale 8 --model-size tiny --out results/perframe/smoke`
Expected: writes `results/perframe/smoke/{config.json,scores.csv,montages/1400.png}`, prints the frame's own_coverage / total_foreign / overlap. Confirm the montage looks sane.

- [ ] **Step 6: Commit**

```bash
git add run_perframe.py tests/test_run_perframe_smoke.py
git commit -m "feat(run_perframe): Approach 1 prompt-based per-frame segmentation"
```

---

### Task 6: Eval sweep + experiments log + docs

**Files:**
- Modify: `run_perframe.py` (a `--sweep` mode that loops the Approach-1 knob combinations)
- Create: `docs/explanation/perframe-experiments.md`
- Modify: `docs/reference/cli.md`, `.gitignore` (ignore `results/`)

**Interfaces:**
- Consumes: Task 5's runner.
- Produces: a documented A/B over Approach-1 knobs on the frame sample.

- [ ] **Step 1: Add `.gitignore` entry + the experiments log skeleton**

Add `results/` to `.gitignore`. Create `docs/explanation/perframe-experiments.md` with a header explaining the results layout and an empty run table (columns: run, approach, negatives, selection, resolver, frames, own_coverage, total_foreign, mean_boundary_on_membrane, overlap_fraction, notes).

- [ ] **Step 2: Add the sweep loop**

In `run_perframe.py`, add `--sweep` that runs Approach 1 over the knob grid `negatives{on,off} x selection{pred_iou,generous,metric} x resolver{argmax,watershed}` on the given `--frames`, each to `results/perframe/<auto-name>/`, and appends one summary row per run to `docs/explanation/perframe-experiments.md`. Keep it a thin loop over `segment_frame_prompt`.

- [ ] **Step 3: CPU smoke the sweep on one frame**

Run: `py -3 run_perframe.py --approach prompt --sweep --frames 1400 --scale 8 --model-size tiny --out results/perframe/sweep_smoke`
Expected: 12 result dirs (2x3x2), 12 rows appended to the experiments log, no crash.

- [ ] **Step 4: Docs + verify**

Update `docs/reference/cli.md` with the `run_perframe.py` flags. Run the `humanizer` skill on the experiments-log header prose. Run `py -3 -m pytest -q && ruff check .`.

- [ ] **Step 5: Commit**

```bash
git add run_perframe.py docs/explanation/perframe-experiments.md docs/reference/cli.md .gitignore
git commit -m "feat(run_perframe): Approach-1 knob sweep + experiments log + docs"
```

---

## Self-review notes

Spec coverage: F1 (T1), F3 both resolvers (T2), selector + matcher (T3), F2 (T4), Approach 1 + negatives + selection knobs (T5), sweep + docs/results log (T6). Approach 2 + tuner are Plan 2. The matcher (T3) is built here because it is a pure primitive Plan 2 consumes. Watershed uses `skimage.segmentation.watershed` (skimage is already a dependency).
