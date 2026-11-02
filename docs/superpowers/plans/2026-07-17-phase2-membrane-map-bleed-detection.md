# Phase 2 foundation: membrane map + bleed detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a ground-truth-free membrane signal and three membrane-aware detector scalars that extend the Phase-0 merge-metric to catch mild bleed and underfill.

**Architecture:** A pure `membrane_map(em_patch)` ridge filter plus three pure detector primitives live in a new library module `sam2_utils/membrane.py`. A `MembraneSource` in `eval/` reads the raw EM per z (via the existing `pipeline.load_frame_sam` seam), crops to each mask's `_sam` window, and feeds the detectors. `eval/merge_metric.py` gains four per-frame columns and new summary quantities, headlined by `mild_bleed_rate`. Nothing changes a mask; this is measurement only.

**Tech Stack:** Python, numpy, scipy.ndimage, scikit-image (`skimage.filters.sato`), pandas. No torch (the module and its tests stay CPU-only).

## Global Constraints

- No em dashes anywhere (code, comments, docstrings, commit messages). Use commas, colons, parentheses, or separate sentences.
- Tests are CPU-only and torch-free: `py -3 -m pytest`.
- Lint with `ruff check .`; only clean files you touch, do not reformat the tree.
- The library (`pipeline.py`, `sam2_utils/`) must never import the drivers (`batch`, `gui`, `run_aval`) or `eval`. `tests/test_import_direction.py` enforces this. The new `sam2_utils/membrane.py` may import numpy/scipy/skimage only.
- Run the `humanizer` skill on any prose before committing (docstrings that are prose, docs, ADR, commit messages).
- v1 defaults are resolution-aware for the `_sam` grid (scale ~8) and comparative across runs, not absolute: `tau=0.5`, `f=0.15`, `tol=2`, `k=6`, `sigmas=(1,2,3)`.

---

### Task 1: The membrane map (ridge filter)

**Files:**
- Create: `sam2_utils/membrane.py`
- Test: `tests/test_membrane_metric.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `membrane_map(em_patch: np.ndarray, *, sigmas=(1,2,3)) -> np.ndarray` returning a float32 map in [0, 1], same H x W as the input (accepts grayscale 2D or RGB 3D). Module constants `DEFAULT_SIGMAS`, `DEFAULT_TAU`, `DEFAULT_F`, `DEFAULT_TOL`, `DEFAULT_K`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_membrane_metric.py
import numpy as np
from sam2_utils import membrane as mb


def test_membrane_map_responds_on_dark_ridge():
    patch = np.full((24, 24), 200, dtype=np.uint8)
    patch[:, 11:13] = 20  # a dark vertical ridge (membranes are dark)
    m = mb.membrane_map(patch)
    assert m.shape == (24, 24)
    assert m.dtype == np.float32
    assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0
    assert m[:, 10:14].mean() > m[:, 0:3].mean()  # ridge lights up vs flat area
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_membrane_metric.py::test_membrane_map_responds_on_dark_ridge -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sam2_utils.membrane'`.

- [ ] **Step 3: Write minimal implementation**

```python
# sam2_utils/membrane.py
"""Membrane / boundary signal for the target worm (roadmap Phase 2, foundation).

A ground-truth-free per-pixel membrane-ness map read from the raw EM, plus the
pure detector primitives that grade a mask against it. The map generator is v1
(a classical dark-ridge filter); the signature is the interface, so a trained
model can drop in behind membrane_map() later without touching the detectors or
the eval scorer. Design:
docs/superpowers/specs/2026-07-17-phase2-membrane-map-bleed-detection-design.md
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# v1 defaults, resolution-aware for the _sam grid (scale ~8). Comparative, not absolute.
DEFAULT_SIGMAS = (1, 2, 3)
DEFAULT_TAU = 0.5   # membrane threshold on the normalised [0, 1] map
DEFAULT_F = 0.15    # min component area as a fraction of the mask, for spanning
DEFAULT_TOL = 2     # px tolerance for boundary-on-membrane
DEFAULT_K = 6       # px flood radius for underfill


def membrane_map(em_patch: np.ndarray, *, sigmas=DEFAULT_SIGMAS) -> np.ndarray:
    """Per-pixel membrane-ness in [0, 1] for a grayscale or RGB EM patch.

    v1: a Sato dark-ridge filter (membranes are dark on bright cytoplasm),
    normalised by its 99th percentile so tau is stable across frames. Returns
    float32, same H x W as the input.
    """
    from skimage.filters import sato

    img = em_patch
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    resp = sato(img, sigmas=sigmas, black_ridges=True).astype(np.float32)
    denom = float(np.percentile(resp, 99)) + 1e-6
    return np.clip(resp / denom, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_membrane_metric.py::test_membrane_map_responds_on_dark_ridge -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/membrane.py tests/test_membrane_metric.py
git commit -m "feat(membrane): v1 dark-ridge membrane map (Phase 2 2a)"
```

---

### Task 2: Interior spanning-membrane detector (primary, soma-safe)

**Files:**
- Modify: `sam2_utils/membrane.py`
- Test: `tests/test_membrane_metric.py`

**Interfaces:**
- Consumes: module constants from Task 1.
- Produces: `spanning_membrane(mask: np.ndarray, mem: np.ndarray, *, tau=DEFAULT_TAU, f=DEFAULT_F) -> tuple[bool, float]` returning `(spanning_merge, bled_fraction)`; and the private helper `_perimeter(mask: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_membrane_metric.py

def _rect_mask(h=30, w=30, y0=5, y1=25, x0=5, x1=25):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_spanning_membrane_flags_ridge_across_mask():
    mask = _rect_mask()
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[:, 14:16] = 1.0  # a ridge cutting the mask border-to-border
    spanning, frac = mb.spanning_membrane(mask, mem)
    assert spanning is True
    assert 0.0 < frac <= 0.5


def test_spanning_membrane_ignores_nucleus_loop():
    mask = _rect_mask()
    mem = np.zeros((30, 30), dtype=np.float32)
    # a closed loop well inside the mask (a nucleus), touching no mask border
    mem[10:20, 10] = 1.0; mem[10:20, 19] = 1.0
    mem[10, 10:20] = 1.0; mem[19, 10:20] = 1.0
    spanning, frac = mb.spanning_membrane(mask, mem)
    assert spanning is False
    assert frac == 0.0


def test_spanning_membrane_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    assert mb.spanning_membrane(mask, np.zeros((30, 30), np.float32)) == (False, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_membrane_metric.py -k spanning -v`
Expected: FAIL with `AttributeError: module 'sam2_utils.membrane' has no attribute 'spanning_membrane'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sam2_utils/membrane.py

def _perimeter(mask: np.ndarray) -> np.ndarray:
    """The 1-px inner boundary ring of a boolean mask."""
    return mask & ~ndi.binary_erosion(mask)


def spanning_membrane(mask: np.ndarray, mem: np.ndarray, *,
                      tau: float = DEFAULT_TAU, f: float = DEFAULT_F
                      ) -> tuple[bool, float]:
    """Detect a membrane ridge that spans the mask border-to-border.

    Remove membrane (mem > tau) from the mask, label the remainder, keep
    components with area >= f * area(mask). If two or more kept components each
    touch the mask's outer border, a membrane cut the mask in two: it engulfed a
    cell boundary. Returns (spanning_merge, bled_fraction), bled_fraction being
    the second-largest border-touching component area / mask area.

    A nucleus (a closed interior loop) leaves one border-touching cytoplasm
    region plus one enclosed region that does not touch the border, so a soma is
    not flagged, by construction.
    """
    area = int(mask.sum())
    if area == 0:
        return False, 0.0
    opened = mask & (mem <= tau)
    lbl, n = ndi.label(opened)
    if n == 0:
        return False, 0.0
    perim = _perimeter(mask)
    min_area = f * area
    border_areas: list[int] = []
    for i in range(1, n + 1):
        comp = lbl == i
        a = int(comp.sum())
        if a < min_area:
            continue
        if bool((comp & perim).any()):
            border_areas.append(a)
    border_areas.sort(reverse=True)
    if len(border_areas) >= 2:
        return True, border_areas[1] / area
    return False, 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_membrane_metric.py -k spanning -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/membrane.py tests/test_membrane_metric.py
git commit -m "feat(membrane): spanning-membrane merge detector, soma-safe (Phase 2 2b-A)"
```

---

### Task 3: Boundary-on-membrane and underfill detectors

**Files:**
- Modify: `sam2_utils/membrane.py`
- Test: `tests/test_membrane_metric.py`

**Interfaces:**
- Consumes: `_perimeter`, module constants.
- Produces: `boundary_on_membrane(mask, mem, *, tau=DEFAULT_TAU, tol=DEFAULT_TOL) -> float`; `underfill_fraction(mask, mem, *, tau=DEFAULT_TAU, k=DEFAULT_K) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_membrane_metric.py

def test_boundary_on_membrane_high_when_edge_on_ridge():
    mask = _rect_mask()  # perimeter at rows/cols 5 and 24
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[4:26, 4:26] = 0.0
    mem[5, 5:25] = 1.0; mem[24, 5:25] = 1.0     # ridge along top+bottom edges
    mem[5:25, 5] = 1.0; mem[5:25, 24] = 1.0     # ridge along left+right edges
    on = mb.boundary_on_membrane(mask, mem)
    assert on > 0.8
    assert mb.boundary_on_membrane(mask, np.zeros((30, 30), np.float32)) == 0.0


def test_underfill_high_when_mask_inset_from_membrane_box():
    mem = np.zeros((30, 30), dtype=np.float32)
    mem[5, 5:25] = 1.0; mem[24, 5:25] = 1.0     # a membrane box the cell lives in
    mem[5:25, 5] = 1.0; mem[5:25, 24] = 1.0
    inset = _rect_mask(y0=10, y1=15, x0=10, x1=15)   # small, room to grow
    filled = _rect_mask(y0=6, y1=24, x0=6, x1=24)    # fills the box
    assert mb.underfill_fraction(inset, mem, k=10) > 0.5
    assert mb.underfill_fraction(filled, mem, k=10) < 0.2


def test_underfill_empty_mask():
    assert mb.underfill_fraction(np.zeros((10, 10), bool), np.zeros((10, 10), np.float32)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_membrane_metric.py -k "boundary or underfill" -v`
Expected: FAIL with `AttributeError` for `boundary_on_membrane` / `underfill_fraction`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to sam2_utils/membrane.py

def boundary_on_membrane(mask: np.ndarray, mem: np.ndarray, *,
                         tau: float = DEFAULT_TAU, tol: int = DEFAULT_TOL) -> float:
    """Fraction of the mask perimeter within tol px of a membrane pixel. Low
    means the edge floats through cytoplasm (leaking bleed or underfill)."""
    perim = _perimeter(mask)
    p = int(perim.sum())
    if p == 0:
        return 0.0
    memb = mem > tau
    if tol > 0:
        memb = ndi.binary_dilation(memb, iterations=tol)
    return float((perim & memb).sum()) / p


def underfill_fraction(mask: np.ndarray, mem: np.ndarray, *,
                       tau: float = DEFAULT_TAU, k: int = DEFAULT_K) -> float:
    """k-bounded flood out of the mask into cytoplasm (mem <= tau), membranes as
    walls. Returns reachable cytoplasm area outside the mask / mask area: high
    means the mask stopped short of its enclosing membrane (room to grow).

    Lowest-confidence of the three detectors: at coarse _sam a broken ridge lets
    the flood leak into a neighbour and overestimate. The k bound keeps a leak
    local. Measured only, never applied (refinement is a separate spec)."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    cyto = mem <= tau
    reach = mask.copy()
    for _ in range(int(k)):
        grown = (ndi.binary_dilation(reach) & cyto) | mask
        if int(grown.sum()) == int(reach.sum()):
            break
        reach = grown
    return float((reach & ~mask).sum()) / area
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_membrane_metric.py -k "boundary or underfill" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/membrane.py tests/test_membrane_metric.py
git commit -m "feat(membrane): boundary-on-membrane + underfill detectors (Phase 2 2b-B/C)"
```

---

### Task 4: MembraneSource (EM patch loader, graceful degradation)

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: `pipeline.load_frame_sam(catmaid_z, *, scale, frame_store=None) -> (image RGB uint8, (H, W))`; `sam2_utils.membrane.membrane_map`.
- Produces: `class MembraneSource(scale: int, *, sigmas=membrane.DEFAULT_SIGMAS, frame_store=None)` with `map_for(z: int, x0: int, y0: int, h: int, w: int) -> np.ndarray | None`. Returns None when the EM for z is unavailable or the window is out of bounds.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_merge_metric.py
from sam2_utils import membrane as mb


def test_membrane_source_crops_and_maps(monkeypatch):
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    frame[:, 20:22] = 20  # a dark ridge in the full _sam frame
    monkeypatch.setattr(mm.pipeline, "load_frame_sam",
                        lambda z, *, scale, frame_store=None: (frame, (0, 0)))
    src = mm.MembraneSource(scale=8)
    m = src.map_for(1400, x0=10, y0=10, h=20, w=20)
    assert m is not None and m.shape == (20, 20)
    assert float(m.max()) <= 1.0


def test_membrane_source_missing_frame_returns_none(monkeypatch):
    def boom(z, *, scale, frame_store=None):
        raise FileNotFoundError(z)
    monkeypatch.setattr(mm.pipeline, "load_frame_sam", boom)
    src = mm.MembraneSource(scale=8)
    assert src.map_for(1400, 0, 0, 10, 10) is None


def test_membrane_source_out_of_bounds_returns_none(monkeypatch):
    frame = np.full((30, 30, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(mm.pipeline, "load_frame_sam",
                        lambda z, *, scale, frame_store=None: (frame, (0, 0)))
    src = mm.MembraneSource(scale=8)
    assert src.map_for(1400, x0=25, y0=25, h=20, w=20) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_merge_metric.py -k membrane_source -v`
Expected: FAIL with `AttributeError: module 'eval.merge_metric' has no attribute 'MembraneSource'`.

- [ ] **Step 3: Write minimal implementation**

```python
# in eval/merge_metric.py: add to the imports at the top
from sam2_utils import alignment, config, membrane

# add after the imports / DEFAULT_RADIUS

class MembraneSource:
    """Supplies membrane maps for a run's masks, cropped to each mask's _sam
    window. Reads the raw EM per z via pipeline.load_frame_sam (the FrameStore
    seam), caches the grayscale _sam frame per z, and runs membrane_map on the
    window. Returns None when the EM for z is unavailable or the window is out
    of bounds, so the scorer degrades to the Phase-0 (node-only) metric."""

    def __init__(self, scale: int, *, sigmas=membrane.DEFAULT_SIGMAS, frame_store=None):
        self.scale = int(scale)
        self.sigmas = sigmas
        self.frame_store = frame_store
        self._gray: dict[int, np.ndarray | None] = {}

    def _frame_gray(self, z: int):
        if z in self._gray:
            return self._gray[z]
        try:
            img, _ = pipeline.load_frame_sam(
                int(z), scale=self.scale, frame_store=self.frame_store)
            gray = (img.mean(axis=2) if img.ndim == 3 else img).astype(np.float32)
        except Exception:
            gray = None
        self._gray[z] = gray
        return gray

    def map_for(self, z: int, x0: int, y0: int, h: int, w: int):
        gray = self._frame_gray(int(z))
        if gray is None:
            return None
        H, W = gray.shape[:2]
        if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            return None
        crop = gray[y0:y0 + h, x0:x0 + w]
        if crop.size == 0 or crop.shape != (h, w):
            return None
        return membrane.membrane_map(crop, sigmas=self.sigmas)
```

Note: `merge_metric.py` already does `import pipeline` and `from sam2_utils import alignment, config`; extend that second import to include `membrane` (shown above) rather than adding a duplicate import line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_merge_metric.py -k membrane_source -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): MembraneSource EM-patch loader with graceful degradation (Phase 2)"
```

---

### Task 5: Wire detectors into the merge-metric (columns, summary, CLI)

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: `MembraneSource` (Task 4); `membrane.spanning_membrane`, `membrane.boundary_on_membrane`, `membrane.underfill_fraction` (Tasks 2 and 3).
- Produces: `score_chain(chain_dir, neuron, nodes_by_z, radius, membrane_source=None, tau=membrane.DEFAULT_TAU, tol=membrane.DEFAULT_TOL)`; `score_run(root, annotate_df=None, radius=DEFAULT_RADIUS, membrane_source="auto", tau=membrane.DEFAULT_TAU, tol=membrane.DEFAULT_TOL)`; new per-frame columns `spanning_merge`, `bled_fraction`, `boundary_on_membrane`, `underfill_fraction`; new summary keys `mild_bleed_rate`, `spanning_merge_rate`, `mean_boundary_on_membrane`, `mean_underfill_fraction`; CLI flags `--no-membrane`, `--tau`, `--tol` (threaded to the detectors).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_merge_metric.py

class _StubSource:
    """Membrane map with a single vertical ridge at column `ridge_x` (full frame)."""
    def __init__(self, ridge_x=15, shape=(50, 50)):
        self.ridge_x, self.shape = ridge_x, shape
    def map_for(self, z, x0, y0, h, w):
        m = np.zeros((h, w), dtype=np.float32)
        col = self.ridge_x - x0
        if 0 <= col < w:
            m[:, col] = 1.0
        return m


def test_score_chain_adds_membrane_columns(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1  # spans ridge at x=15
    d = _write_chain(tmp_path, "AVAL_chain00", {1400: a})
    nbz = {1400: [(15.0, 15.0, "AVAL", "own0")]}  # own node, no foreign
    rec = mm.score_chain(d, "AVAL", nbz, radius=0, membrane_source=_StubSource(15))[0]
    assert rec["spanning_merge"] is True
    assert rec["bled_fraction"] > 0.0
    assert 0.0 <= rec["boundary_on_membrane"] <= 1.0


def test_score_run_reports_mild_bleed_rate(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1  # spans ridge, no foreign node
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({"node_id": ["own0"], "cell_name": ["AVAL"],
                       "z": [1400], "x_tif": [120.0], "y_tif": [120.0]})
    per, summ = mm.score_run(root, annotate_df=df, radius=0, membrane_source=_StubSource(15))
    assert summ["total_foreign_nodes"] == 0
    assert summ["mild_bleed_rate"] == 1.0          # spanning merge with no foreign node
    assert summ["spanning_merge_rate"] == 1.0
    assert "spanning_merge" in per.columns


def test_score_run_no_membrane_keeps_phase0(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({"node_id": ["own0"], "cell_name": ["AVAL"],
                       "z": [1400], "x_tif": [120.0], "y_tif": [120.0]})
    per, summ = mm.score_run(root, annotate_df=df, radius=0, membrane_source=None)
    assert summ["mild_bleed_rate"] is None
    assert per["spanning_merge"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_merge_metric.py -k "membrane_columns or mild_bleed or no_membrane" -v`
Expected: FAIL (score_chain rejects the `membrane_source` kwarg / new keys missing).

- [ ] **Step 3: Write minimal implementation**

In `score_chain`, add the parameter and compute the membrane scalars per frame. Replace the current signature and record construction:

```python
def score_chain(chain_dir: Path, neuron: str,
                nodes_by_z: dict[int, list[tuple[float, float, str, str]]],
                radius: int, membrane_source=None,
                tau: float = membrane.DEFAULT_TAU,
                tol: int = membrane.DEFAULT_TOL) -> list[dict]:
    """Per-z merge/dropout records for one chain, from its RAW saved masks.

    When membrane_source is given, each record also carries the membrane-aware
    detector scalars (spanning_merge, bled_fraction, boundary_on_membrane,
    underfill_fraction); they are None when the source has no map for that
    frame."""
    masks = pipeline.chain_masks_in_sam(Path(chain_dir))
    recs: list[dict] = []
    for z, (mask, x0, y0) in sorted(masks.items()):
        nodes = nodes_by_z.get(int(z), [])
        own = [(x, y) for (x, y, cell, _nid) in nodes if cell == neuron]
        own_ok = any(own_contained(mask, x0, y0, xy, radius) for xy in own) if own else False
        fids = foreign_hits(mask, x0, y0, nodes, neuron, radius)
        rec = {
            "z": int(z),
            "own_contained": bool(own_ok),
            "n_foreign": len(fids),
            "foreign_ids": fids,
            "empty": bool(not mask.any()),
            "spanning_merge": None,
            "bled_fraction": None,
            "boundary_on_membrane": None,
            "underfill_fraction": None,
        }
        if membrane_source is not None:
            h, w = mask.shape[:2]
            mem = membrane_source.map_for(int(z), int(x0), int(y0), h, w)
            if mem is not None:
                spanning, frac = membrane.spanning_membrane(mask, mem, tau=tau)
                rec["spanning_merge"] = bool(spanning)
                rec["bled_fraction"] = float(frac)
                rec["boundary_on_membrane"] = float(
                    membrane.boundary_on_membrane(mask, mem, tau=tau, tol=tol))
                rec["underfill_fraction"] = float(
                    membrane.underfill_fraction(mask, mem, tau=tau))
        recs.append(rec)
    return recs
```

In `score_run`, build the source (unless disabled), thread it through, and add the summary keys. Change the signature and the body:

```python
def score_run(root, annotate_df: pd.DataFrame | None = None,
              radius: int = DEFAULT_RADIUS, membrane_source="auto",
              tau: float = membrane.DEFAULT_TAU, tol: int = membrane.DEFAULT_TOL
              ) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-chain records, write CSV, return per-frame DataFrame and summary.

    membrane_source: "auto" builds a MembraneSource for the run scale; None
    disables the membrane pass (Phase-0-only); or pass an object with map_for()
    for tests. When membrane scalars are absent, the membrane summary keys are
    None and the Phase-0 keys are unchanged."""
    root = Path(root)
    scale = run_scale(root)
    if annotate_df is None:
        annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, scale)
    if membrane_source == "auto":
        membrane_source = MembraneSource(scale)

    rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius,
                                   membrane_source, tau=tau, tol=tol):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)

    per = pd.DataFrame(rows)
    n_frames = len(per)
    have_mem = bool(n_frames) and per["spanning_merge"].notna().any()
    summary = {
        "n_chains": int(per[["neuron", "chain_idx"]].drop_duplicates().shape[0]) if n_frames else 0,
        "n_frames": int(n_frames),
        "foreign_frame_rate": float((per["n_foreign"] > 0).mean()) if n_frames else 0.0,
        "dropout_rate": float((per["empty"] | ~per["own_contained"]).mean()) if n_frames else 0.0,
        "total_foreign_nodes": int(per["n_foreign"].sum()) if n_frames else 0,
        "mild_bleed_rate": None,
        "spanning_merge_rate": None,
        "mean_boundary_on_membrane": None,
        "mean_underfill_fraction": None,
    }
    if have_mem:
        scored = per[per["spanning_merge"].notna()]
        span = scored["spanning_merge"].astype(bool)
        summary["spanning_merge_rate"] = float(span.mean())
        summary["mild_bleed_rate"] = float((span & (scored["n_foreign"] == 0)).mean())
        summary["mean_boundary_on_membrane"] = float(scored["boundary_on_membrane"].mean())
        summary["mean_underfill_fraction"] = float(scored["underfill_fraction"].mean())
    if n_frames:
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        per_out.to_csv(root / "_merge_metric.csv", index=False)
    return per, summary
```

Extend `format_summary` to append a membrane line when present:

```python
def format_summary(name: str, s: dict) -> str:
    line = (f"{name:<28} chains={s['n_chains']:>4} frames={s['n_frames']:>6} "
            f"foreign_frame_rate={s['foreign_frame_rate']:.3f} "
            f"dropout_rate={s['dropout_rate']:.3f} "
            f"total_foreign={s['total_foreign_nodes']:>5}")
    if s.get("mild_bleed_rate") is not None:
        line += (f" | mild_bleed_rate={s['mild_bleed_rate']:.3f} "
                 f"spanning_merge_rate={s['spanning_merge_rate']:.3f} "
                 f"boundary_on_membrane={s['mean_boundary_on_membrane']:.3f} "
                 f"underfill={s['mean_underfill_fraction']:.3f}")
    return line
```

Add the CLI flags in `main` and pass them through (build the source explicitly so `--no-membrane` and the params take effect):

```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Target-worm skeleton merge-metric (roadmap Phase 0 + Phase 2).")
    ap.add_argument("--root", action="append", required=True, dest="roots",
                    help="a merged run tree; repeat to compare runs")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    ap.add_argument("--no-membrane", action="store_true",
                    help="skip the Phase-2 membrane detectors (Phase-0-only, no EM reads)")
    ap.add_argument("--tau", type=float, default=membrane.DEFAULT_TAU,
                    help="membrane threshold on the normalised [0,1] map")
    ap.add_argument("--tol", type=int, default=membrane.DEFAULT_TOL,
                    help="px tolerance for boundary-on-membrane")
    args = ap.parse_args(argv)

    annotate_df = load_node_table()
    for root in args.roots:
        src = None if args.no_membrane else MembraneSource(run_scale(root))
        _per, summ = score_run(root, annotate_df=annotate_df, radius=args.radius,
                               membrane_source=src, tau=args.tau, tol=args.tol)
        print(format_summary(Path(root).name, summ))
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_merge_metric.py -v`
Expected: PASS (all existing tests plus the three new ones). The existing `test_score_run_aggregates` still passes because it passes `annotate_df` and relies on `membrane_source="auto"`, but that run tree has no real EM; `MembraneSource.map_for` returns None (load_frame_sam raises for the fake z), so membrane columns are NaN and the Phase-0 assertions are unchanged.

- [ ] **Step 5: Run the full suite and lint**

Run: `py -3 -m pytest -q && ruff check eval/merge_metric.py sam2_utils/membrane.py tests/test_membrane_metric.py tests/test_merge_metric.py`
Expected: all green, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): membrane-aware bleed detection in merge-metric, mild_bleed_rate headline (Phase 2 2b)"
```

---

### Task 6: Documentation, ADR, roadmap, CHANGELOG

**Files:**
- Modify: `docs/reference/configuration.md`, `docs/reference/cli.md`, `docs/reference/code-map.md`, `docs/explanation/roadmap.md`, `docs/CHANGELOG.md`
- Create: `docs/adr/0016-membrane-map-border-to-border-bleed-detection.md`

**Interfaces:**
- Consumes: the landed code from Tasks 1 to 5.
- Produces: documentation only, no code.

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0016-membrane-map-border-to-border-bleed-detection.md`. Cover: context (Phase-0 merge-metric is a severe-merge floor, blind to mild bleed and underfill); decision (a v1 classical ridge-filter membrane map behind a swappable interface, plus three detector scalars; the border-to-border spanning criterion is what makes the soma case fall out for free); why border-to-border rather than "any membrane inside the mask" (a nucleus is a closed interior loop, not a spanning ridge, so it must not count); consequences (comparative not absolute at coarse `_sam`; underfill is lowest-confidence; refinement 2c and arbitration 2d are deferred and will reuse this signal). Follow the format of an existing ADR (read `docs/adr/0015-target-worm-merge-metric-ruler.md` first). No em dashes.

- [ ] **Step 2: Update the reference docs**

- `docs/reference/configuration.md`: add the membrane metric parameters (`tau`, `f`, `tol`, `k`, `sigmas`) with their v1 defaults and the note that they are comparative, resolution-aware for the `_sam` grid.
- `docs/reference/cli.md`: add the `merge_metric` flags `--no-membrane`, `--tau`, `--tol`; note the new summary line (`mild_bleed_rate`, `spanning_merge_rate`, `boundary_on_membrane`, `underfill`) and the four new `_merge_metric.csv` columns.
- `docs/reference/code-map.md`: add `sam2_utils/membrane.py` as the home of the membrane signal and the detector primitives, and note `eval/merge_metric.py` now owns `MembraneSource` and the membrane-aware scoring.

- [ ] **Step 3: Update roadmap and CHANGELOG**

- `docs/explanation/roadmap.md`: in the Phase 2 section and the §5b immediate queue, mark the foundation (2a membrane map + 2b detection) as landed, and note that 2c grow-to-membrane refinement and 2d non-overlap arbitration remain queued as their own specs. Keep it forward-looking; move the build detail to the CHANGELOG.
- `docs/CHANGELOG.md`: add an entry for the Phase 2 foundation (membrane map + membrane-aware bleed detection, `mild_bleed_rate`), referencing the spec and ADR 0016.

- [ ] **Step 4: Humanize and verify**

Run the `humanizer` skill on the ADR and CHANGELOG prose. Then:

Run: `py -3 -m pytest -q && ruff check .`
Expected: green and clean.
Manually confirm no em dashes in the touched docs and that internal markdown links resolve.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "docs: Phase 2 foundation (membrane map + bleed detection), ADR 0016, roadmap, CHANGELOG"
```

---

## Post-plan: retro-score the Phase-1 A/B trees

Not a code task, the payoff. Once Task 5 lands, run the membrane-aware metric on the four existing trees to finally grade their mild bleed and underfill (needs the raw EM reachable via `config.WORM_PATH` on F:):

```bash
py -3 -m eval.merge_metric \
  --root <...>/original_tier2_s1forced_neg_merged \
  --root <...>/original_generous_only_merged \
  --root <...>/original_perslice_only_merged \
  --root <...>/original_perslice_merged
```

Compare `mild_bleed_rate` and `mean_underfill_fraction` across the four. This tests the standing hypothesis: per-slice wins on severe merge (Phase-0) but may trade it for mild bleed on its blow-up tail, and generous may show up as underfill-reducing but mild-bleed-increasing. Record the verdict in the CHANGELOG.