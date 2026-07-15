# Target-worm skeleton merge-metric (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ground-truth-free scorer that grades a run's raw per-chain masks for severe bleed (a mask containing another neuron's skeleton node) and dropout (a mask missing its own node), plus a CLI to retro-score the existing merged runs.

**Architecture:** A new `eval/merge_metric.py`. It reads the CATMAID node table once (`cell_name`, `z`, `x_tif`, `y_tif`), reads each chain's raw masks via `pipeline.chain_masks_in_sam` (which already remaps tier-2 `_pcrop` masks onto the `_sam` grid), samples nodes onto that same grid using the run's `scale` from `_run_meta.json`, and counts foreign-node hits and own-node dropout per z. It aggregates per neuron and per run into a CSV plus a printed summary. `eval` may import `pipeline`; never the reverse, so the metric lives in `eval/`.

**Tech Stack:** Python, numpy, pandas, OpenCV (via `pipeline`), pytest (CPU-only, torch-free).

## Global Constraints

- No em dashes anywhere in code, comments, or commit messages. Use commas, colons, parentheses, or separate sentences.
- Tests are CPU-only and torch-free: `py -3 -m pytest`.
- `ruff check .` must stay clean; only touch files this plan touches.
- Import direction: the library (`pipeline`, `sam2_utils`) must never import `eval`. `eval` may import `pipeline`. This metric goes in `eval/`.
- Score the **raw per-chain masks** via `pipeline.chain_masks_in_sam`, never the composite / non-overlap labelmap (the non-overlap argmax would zero the signal by construction).
- This metric is a **severe-merge floor** (it only catches bleed that reaches a foreign centreline). Do not compute or report ERL here; that is out of scope by design (see roadmap §5 Phase 0).
- Node-to-grid divisor is the run's `scale` (from `_run_meta.json` `resolution.scale`), matching `chain_masks_in_sam`'s `sam_scale`. All current presets have `scale == save_downscale`; if a run's `_run_meta.json` shows them unequal, the scorer must warn and skip.

---

### Task 1: Node table loader and per-z grid sampling

**Files:**
- Create: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: `sam2_utils.config.CSV_PATH`, `sam2_utils.alignment.catmaid_to_tif`.
- Produces:
  - `load_node_table() -> pandas.DataFrame` with columns including `cell_name`, `z`, `x_tif`, `y_tif`.
  - `nodes_by_z(annotate_df: pandas.DataFrame, scale: int) -> dict[int, list[tuple[float, float, str, str]]]` returning `{z: [(x_grid, y_grid, cell_name, node_id), ...]}` where `x_grid = x_tif / scale`, `y_grid = y_tif / scale`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_metric.py
import pandas as pd
from eval import merge_metric as mm

def test_nodes_by_z_groups_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"],
        "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401],
        "x_tif": [800.0, 1600.0, 240.0],
        "y_tif": [80.0, 160.0, 800.0],
    })
    got = mm.nodes_by_z(df, scale=8)
    assert set(got) == {1400, 1401}
    assert sorted(got[1400]) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert got[1401] == [(30.0, 100.0, "AVAL", "c")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_nodes_by_z_groups_and_scales -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError: module 'eval.merge_metric' has no attribute 'nodes_by_z'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/merge_metric.py
"""Target-worm skeleton merge-metric (roadmap Phase 0).

A ground-truth-free severe-bleed / dropout scorer for a run's RAW per-chain masks,
scored against the target worm's own CATMAID skeletons. See docs/explanation/roadmap.md
section 5 Phase 0 for the scope: this is a severe-merge floor, not an ERL benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sam2_utils import alignment, config


def load_node_table() -> pd.DataFrame:
    """The CATMAID node table with x_tif / y_tif attached (mirrors batch.py setup)."""
    df = pd.read_csv(config.CSV_PATH)
    xy_tif = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
    df["x_tif"] = xy_tif[:, 0]
    df["y_tif"] = xy_tif[:, 1]
    return df


def nodes_by_z(annotate_df: pd.DataFrame, scale: int
               ) -> dict[int, list[tuple[float, float, str, str]]]:
    """Group nodes by z with coordinates on the run's _sam grid (x_tif / scale).

    Matches chain_masks_in_sam's grid so a node maps into a returned mask directly.
    """
    out: dict[int, list[tuple[float, float, str, str]]] = {}
    for z, x_tif, y_tif, cell, nid in zip(
        annotate_df["z"].astype(int), annotate_df["x_tif"].astype(float),
        annotate_df["y_tif"].astype(float), annotate_df["cell_name"].astype(str),
        annotate_df["node_id"].astype(str),
    ):
        out.setdefault(int(z), []).append((x_tif / scale, y_tif / scale, cell, nid))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_nodes_by_z_groups_and_scales -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): merge-metric node table loader + per-z grid sampling"
```

---

### Task 2: Containment counters (own-node + foreign-node)

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: `pipeline._point_in_mask` (re-exported from the `pipeline` package), Task 1 node tuples.
- Produces:
  - `own_contained(mask, x0, y0, node_xy, radius) -> bool`
  - `foreign_hits(mask, x0, y0, nodes, own_neuron, radius) -> list[str]` (returns the `node_id`s of foreign nodes inside the mask). `nodes` is a per-z list of `(x_grid, y_grid, cell_name, node_id)`; `(x0, y0)` is the mask's grid offset from `chain_masks_in_sam`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_metric.py (append)
import numpy as np

def test_containment_own_and_foreign():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True  # a blob at grid (10..19, 10..19)
    nodes = [
        (15.0, 15.0, "AVAL", "own"),    # inside, own neuron
        (14.0, 14.0, "AVAR", "foreign_in"),   # inside, foreign
        (40.0, 40.0, "AVAR", "foreign_out"),  # outside
    ]
    assert mm.own_contained(mask, 0, 0, (15.0, 15.0), radius=0) is True
    assert mm.own_contained(mask, 0, 0, (40.0, 40.0), radius=0) is False
    hits = mm.foreign_hits(mask, 0, 0, nodes, own_neuron="AVAL", radius=0)
    assert hits == ["foreign_in"]

def test_containment_respects_offset():
    mask = np.ones((10, 10), dtype=bool)  # a crop placed at (x0=100, y0=200)
    # a foreign node at grid (105, 205) is local (5, 5): inside
    hits = mm.foreign_hits(mask, 100, 200, [(105.0, 205.0, "X", "n")], own_neuron="AVAL", radius=0)
    assert hits == ["n"]
    # a foreign node at grid (5, 5) is local (-95, -195): outside
    assert mm.foreign_hits(mask, 100, 200, [(5.0, 5.0, "X", "n")], "AVAL", 0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_merge_metric.py -k containment -v`
Expected: FAIL with `AttributeError: ... has no attribute 'own_contained'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/merge_metric.py (append)
import pipeline


def own_contained(mask: np.ndarray, x0: int, y0: int,
                  node_xy: tuple[float, float], radius: int) -> bool:
    """True if the mask covers its own node (grid coords), accounting for the
    mask's (x0, y0) grid offset."""
    x, y = node_xy
    return pipeline._point_in_mask(mask, x - x0, y - y0, radius)


def foreign_hits(mask: np.ndarray, x0: int, y0: int,
                 nodes: list[tuple[float, float, str, str]],
                 own_neuron: str, radius: int) -> list[str]:
    """node_ids of nodes belonging to a DIFFERENT neuron that fall inside the mask."""
    hits: list[str] = []
    for x, y, cell, nid in nodes:
        if cell == own_neuron:
            continue
        if pipeline._point_in_mask(mask, x - x0, y - y0, radius):
            hits.append(nid)
    return hits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_merge_metric.py -k containment -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): own-node and foreign-node containment counters"
```

---

### Task 3: Per-chain scorer

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: `pipeline.chain_masks_in_sam(chain_dir) -> {z: (mask, x0, y0)}`, Task 1 `nodes_by_z`, Task 2 counters.
- Produces:
  - `score_chain(chain_dir, neuron, nodes_by_z, radius) -> list[dict]`, one record per z with keys `z`, `own_contained` (bool), `n_foreign` (int), `foreign_ids` (list[str]), `empty` (bool, mask has no foreground).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_metric.py (append)
import cv2

def _write_chain(tmp_path, name, masks):
    """masks: {z: 2D uint8 array}. Writes a legacy chain (no crop_window)."""
    d = tmp_path / name
    (d / "masks").mkdir(parents=True)
    for z, arr in masks.items():
        cv2.imwrite(str(d / "masks" / f"mask_{z:04d}.png"), (arr > 0).astype("uint8") * 255)
    (d / "state.json").write_text("{}")  # no crop_window -> legacy _sam, offset (0,0)
    return d

def test_score_chain_flags_foreign_and_dropout(tmp_path):
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1   # z1400: covers own+foreign
    b = np.zeros((50, 50), dtype=np.uint8)                        # z1401: empty (dropout)
    d = _write_chain(tmp_path, "AVAL_chain00", {1400: a, 1401: b})
    nbz = {
        1400: [(15.0, 15.0, "AVAL", "own0"), (14.0, 14.0, "AVAR", "f0")],
        1401: [(15.0, 15.0, "AVAL", "own1")],
    }
    recs = {r["z"]: r for r in mm.score_chain(d, "AVAL", nbz, radius=0)}
    assert recs[1400]["own_contained"] and recs[1400]["n_foreign"] == 1
    assert recs[1400]["foreign_ids"] == ["f0"] and not recs[1400]["empty"]
    assert recs[1401]["empty"] and not recs[1401]["own_contained"]
    assert recs[1401]["n_foreign"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_score_chain_flags_foreign_and_dropout -v`
Expected: FAIL with `AttributeError: ... has no attribute 'score_chain'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/merge_metric.py (append)
def score_chain(chain_dir: Path, neuron: str,
                nodes_by_z: dict[int, list[tuple[float, float, str, str]]],
                radius: int) -> list[dict]:
    """Per-z merge/dropout records for one chain, from its RAW saved masks."""
    masks = pipeline.chain_masks_in_sam(Path(chain_dir))
    recs: list[dict] = []
    for z, (mask, x0, y0) in sorted(masks.items()):
        nodes = nodes_by_z.get(int(z), [])
        own = [(x, y) for (x, y, cell, _nid) in nodes if cell == neuron]
        own_ok = any(own_contained(mask, x0, y0, xy, radius) for xy in own) if own else False
        fids = foreign_hits(mask, x0, y0, nodes, neuron, radius)
        recs.append({
            "z": int(z),
            "own_contained": bool(own_ok),
            "n_foreign": len(fids),
            "foreign_ids": fids,
            "empty": bool(not mask.any()),
        })
    return recs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_score_chain_flags_foreign_and_dropout -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): per-chain merge/dropout scorer over raw masks"
```

---

### Task 4: Per-run scorer, aggregation, and CSV output

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: Task 1 `load_node_table` / `nodes_by_z`, Task 3 `score_chain`, `_run_meta.json`.
- Produces:
  - `run_scale(root) -> int` (reads `resolution.scale` from `<root>/_run_meta.json`; raises with a clear message if `scale != save_downscale`).
  - `score_run(root, annotate_df=None, radius=DEFAULT_RADIUS) -> tuple[pandas.DataFrame, dict]`. The DataFrame has one row per (neuron, chain_idx, z); the dict is the run summary: `n_chains`, `n_frames`, `foreign_frame_rate` (fraction of frames with n_foreign > 0), `dropout_rate` (fraction of frames empty or own not contained), `total_foreign_nodes`. Writes `<root>/_merge_metric.csv`.
- Constant: `DEFAULT_RADIUS = 3` (grid px; a small containment tolerance).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_metric.py (append)
import json

def test_score_run_aggregates(tmp_path, monkeypatch):
    # one neuron AVAL with a chain that bleeds onto AVAR's node on one of two frames
    a = np.zeros((50, 50), dtype=np.uint8); a[10:20, 10:20] = 1
    b = np.zeros((50, 50), dtype=np.uint8); b[10:20, 10:20] = 1
    root = tmp_path / "run_merged"
    _write_chain(root / "AVAL", "chain_00", {1400: a, 1401: b})
    (root / "_run_meta.json").write_text(json.dumps(
        {"resolution": {"scale": 8, "save_downscale": 8}}))
    df = pd.DataFrame({
        "node_id": ["own0", "own1", "f0"], "cell_name": ["AVAL", "AVAL", "AVAR"],
        "z": [1400, 1401, 1400], "x_tif": [120.0, 120.0, 112.0], "y_tif": [120.0, 120.0, 112.0],
    })
    per, summ = mm.score_run(root, annotate_df=df, radius=0)
    assert summ["n_chains"] == 1 and summ["n_frames"] == 2
    assert summ["total_foreign_nodes"] == 1        # f0 hit on z1400 only
    assert abs(summ["foreign_frame_rate"] - 0.5) < 1e-9
    assert (root / "_merge_metric.csv").exists()
    assert set(per["neuron"]) == {"AVAL"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_score_run_aggregates -v`
Expected: FAIL with `AttributeError: ... has no attribute 'score_run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/merge_metric.py (append)
import json

DEFAULT_RADIUS = 3


def run_scale(root: Path) -> int:
    meta = json.loads((Path(root) / "_run_meta.json").read_text())
    res = meta.get("resolution", {})
    scale = int(res["scale"])
    sd = int(res.get("save_downscale", scale))
    if sd != scale:
        raise ValueError(
            f"{root}: scale ({scale}) != save_downscale ({sd}); the node grid "
            "assumption does not hold, extend the scorer before trusting it.")
    return scale


def score_run(root, annotate_df: pd.DataFrame | None = None,
              radius: int = DEFAULT_RADIUS) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    scale = run_scale(root)
    if annotate_df is None:
        annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, scale)

    rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)

    per = pd.DataFrame(rows)
    n_frames = len(per)
    summary = {
        "n_chains": int(per[["neuron", "chain_idx"]].drop_duplicates().shape[0]) if n_frames else 0,
        "n_frames": int(n_frames),
        "foreign_frame_rate": float((per["n_foreign"] > 0).mean()) if n_frames else 0.0,
        "dropout_rate": float((per["empty"] | ~per["own_contained"]).mean()) if n_frames else 0.0,
        "total_foreign_nodes": int(per["n_foreign"].sum()) if n_frames else 0,
    }
    if n_frames:
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        per_out.to_csv(root / "_merge_metric.csv", index=False)
    return per, summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_score_run_aggregates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): per-run merge-metric aggregation + CSV output"
```

---

### Task 5: CLI and multi-run comparison

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_merge_metric.py`

**Interfaces:**
- Consumes: Task 4 `score_run`.
- Produces:
  - `format_summary(name, summary) -> str` (one aligned line).
  - `main(argv=None)` for `python -m eval.merge_metric --root A [--root B ...] [--radius N]`, printing one summary line per root (loads the node table once, shared across roots) and writing each root's `_merge_metric.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_metric.py (append)
def test_format_summary_is_one_line():
    s = mm.format_summary("neg", {
        "n_chains": 100, "n_frames": 8052, "foreign_frame_rate": 0.031,
        "dropout_rate": 0.12, "total_foreign_nodes": 274})
    assert "neg" in s and "0.031" in s and "\n" not in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_merge_metric.py::test_format_summary_is_one_line -v`
Expected: FAIL with `AttributeError: ... has no attribute 'format_summary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/merge_metric.py (append)
import argparse


def format_summary(name: str, s: dict) -> str:
    return (f"{name:<28} chains={s['n_chains']:>4} frames={s['n_frames']:>6} "
            f"foreign_frame_rate={s['foreign_frame_rate']:.3f} "
            f"dropout_rate={s['dropout_rate']:.3f} "
            f"total_foreign={s['total_foreign_nodes']:>5}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Target-worm skeleton merge-metric (roadmap Phase 0).")
    ap.add_argument("--root", action="append", required=True, dest="roots",
                    help="a merged run tree; repeat to compare runs")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    args = ap.parse_args(argv)

    annotate_df = load_node_table()
    for root in args.roots:
        _per, summ = score_run(root, annotate_df=annotate_df, radius=args.radius)
        print(format_summary(Path(root).name, summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_merge_metric.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_merge_metric.py
git commit -m "feat(eval): merge-metric CLI + multi-run comparison"
```

---

### Task 6: Retro-score the existing runs and record the verdict

**Files:**
- Modify: `docs/CHANGELOG.md` (record the numbers)

**Interfaces:**
- Consumes: the Task 5 CLI, the five extracted merged trees under `F:\ZhenLab\Data\output_masks\resolution_experiments\`.

- [ ] **Step 1: Run the metric across all runs**

Run (one command, node table loaded once):

```bash
py -3 -m eval.merge_metric \
  --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_fullres_merged" \
  --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_wholeimg_s4_merged" \
  --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_tier2forced_merged" \
  --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_tier2forced_neg_merged" \
  --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_tier2_s1forced_neg_merged"
```

Expected: five summary lines (foreign_frame_rate, dropout_rate, total_foreign per run), and a `_merge_metric.csv` written into each tree.

- [ ] **Step 2: Sanity-check against the known AVAL/ch16 case**

Confirm the metric fires where we already know bleed exists: on `original_fullres_merged`, `AVAL/chain_16` should show foreign hits around z1454-1463 (the documented wrong-cell jump), and on `original_tier2_s1forced_neg_merged` that chain should show far fewer. Inspect the per-run CSV:

Run: `py -3 -c "import pandas as pd; d=pd.read_csv(r'F:/ZhenLab/Data/output_masks/resolution_experiments/original_fullres_merged/_merge_metric.csv'); print(d[(d.neuron=='AVAL')&(d.chain_idx==16)&(d.n_foreign>0)][['z','n_foreign','foreign_ids']].to_string())"`
Expected: rows with n_foreign > 0 in the z1454-1463 band.

- [ ] **Step 3: Record the verdict in the CHANGELOG**

Add a short paragraph under the `r-2026-07-15` entry giving the foreign_frame_rate / dropout_rate for the five runs and the one-line read: does the negatives / full-res second pass actually reduce severe bleed on the target worm (the verdict the flags could not give). No em dashes.

- [ ] **Step 4: Commit**

```bash
git add docs/CHANGELOG.md
git commit -m "docs: record target-worm merge-metric verdict on the resolution/neg runs"
```

---

## Notes for the implementer

- `chain_masks_in_sam` reads `state.json` for the tier-2 `crop_window`; a legacy `_sam` chain has none and its masks come back full-frame at offset `(0, 0)`. The test helper writes `state.json` as `{}` to exercise the legacy path. A tier-2 tree exercises the offset path in real data (Step 1), no separate synthetic tier-2 test is required.
- `pipeline._point_in_mask` is re-exported from the `pipeline` package (`pipeline/__init__.py`), so `import pipeline; pipeline._point_in_mask(...)` works.
- The metric deliberately ignores nodes with no chain in the run and does not need `chains.json`: `cell_name` on the node table is the neuron identity, and a foreign node is simply one whose `cell_name` differs from the chain's neuron (the chain's neuron is its parent directory name in the merged tree).
- `radius` is a small containment tolerance in grid pixels. `DEFAULT_RADIUS = 3` roughly matches the QC skeleton-containment window at scale 8; lower it toward 0 for a stricter merge count, raise it if registration jitter causes false dropouts.
