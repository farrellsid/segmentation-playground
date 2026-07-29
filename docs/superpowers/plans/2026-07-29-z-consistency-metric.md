# Z-to-z consistency metric implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a z-to-z consistency metric (IoU and centroid drift between a chain's consecutive raw
masks) to `eval/merge_metric.py`, so a per-slice-vs-propagation comparison can read a temporal
consistency signal the existing per-frame ruler cannot see.

**Architecture:** Two new pure functions (`z_transitions`, `summarize_z_consistency`) in
`eval/merge_metric.py`, unit-tested against hand-computed values. `score_run` gains a second,
independent pass that loads each chain's masks again (cheap: small PNGs, not EM frames) and folds
the aggregated result into the existing summary dict and a new per-transition CSV.

**Tech Stack:** numpy (already a dependency). No torch, no new dependency.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (`CLAUDE.md`).
- Run the `humanizer` skill on any prose you are about to commit before committing.
- New tests stay torch-free and CPU-only; `py -3 -m pytest`.
- Lint with `ruff check .`, touching only files you edit.
- Do not change `score_chain`'s signature (two existing tests, `tests/test_merge_metric.py:63,167`,
  call it directly with a chain-directory path; this spec's z-consistency pass loads masks
  independently instead of threading them through `score_chain`).
- This is measurement only: no lever, no mask change, no change to which mask a run keeps.
- Commit incrementally, one concern per commit.

---

### Task 1: `z_transitions` and `summarize_z_consistency`

**Files:**
- Modify: `eval/merge_metric.py`
- Test: `tests/test_z_consistency.py` (new)

**Interfaces:**
- Produces: `z_transitions(masks: dict[int, tuple[np.ndarray, int, int]]) -> list[dict]`,
  `summarize_z_consistency(transitions: list[dict], *, low_iou_threshold: float = 0.5) -> dict`
- Consumes: nothing from other tasks (no dependencies).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_z_consistency.py`:

```python
import numpy as np
from eval import merge_metric as mm


def _rect(h=10, w=10, y0=3, y1=7, x0=3, x1=7):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks_perfect_consistency():
    m = _rect()
    masks = {5: (m, 0, 0), 6: (m.copy(), 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["z_from"] == 5 and t["z_to"] == 6 and t["gap"] == 1
    assert t["iou"] == 1.0
    assert t["centroid_drift_px"] == 0.0


def test_shifted_mask_hand_computed_iou_and_drift():
    m_a = _rect()  # x[3:7), y[3:7)
    m_b = _rect(x0=5, x1=9)  # x[5:9), y[3:7), shifted +2 in x
    masks = {5: (m_a, 0, 0), 6: (m_b, 0, 0)}
    t = mm.z_transitions(masks)[0]
    # intersection x[5:7),y[3:7) = 2*4=8; each mask area 4*4=16; union = 16+16-8 = 24
    assert abs(t["iou"] - 8 / 24) < 1e-9
    assert abs(t["centroid_drift_px"] - 2.0) < 1e-9


def test_disjoint_masks_zero_iou():
    m_a = np.zeros((10, 10), dtype=bool); m_a[0:2, 0:2] = True
    m_b = np.zeros((10, 10), dtype=bool); m_b[8:10, 8:10] = True
    masks = {5: (m_a, 0, 0), 6: (m_b, 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["iou"] == 0.0


def test_empty_mask_is_dropout_not_low_consistency():
    m_a = _rect()
    m_empty = np.zeros((10, 10), dtype=bool)
    masks = {5: (m_a, 0, 0), 6: (m_empty, 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["iou"] is None
    assert t["centroid_drift_px"] is None


def test_z_gap_recorded_correctly():
    m = _rect()
    masks = {10: (m, 0, 0), 11: (m.copy(), 0, 0), 13: (m.copy(), 0, 0)}
    ts = mm.z_transitions(masks)
    assert [t["gap"] for t in ts] == [1, 2]


def test_summarize_empty_input():
    s = mm.summarize_z_consistency([])
    assert s["n_transitions"] == 0
    assert s["mean_z2z_iou"] is None
    assert s["mean_centroid_drift_px"] is None
    assert s["frac_gap1_transitions"] is None
    assert s["frac_low_iou"] is None


def test_summarize_aggregates_and_excludes_dropout_and_gaps_correctly():
    transitions = [
        {"gap": 1, "iou": 0.9, "centroid_drift_px": 1.0},
        {"gap": 1, "iou": 0.3, "centroid_drift_px": 5.0},
        {"gap": 1, "iou": None, "centroid_drift_px": None},  # dropout
        {"gap": 2, "iou": 0.4, "centroid_drift_px": 3.0},  # gap != 1
    ]
    s = mm.summarize_z_consistency(transitions)
    assert s["n_transitions"] == 4
    assert s["n_dropout_transitions"] == 1
    assert abs(s["mean_z2z_iou"] - (0.9 + 0.3 + 0.4) / 3) < 1e-9
    assert abs(s["frac_gap1_transitions"] - 3 / 4) < 1e-9
    # frac_low_iou only counts gap==1, scored transitions: [0.9, 0.3], 1 of 2 below 0.5
    assert abs(s["frac_low_iou"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_z_consistency.py -v`
Expected: FAIL, `AttributeError: module 'eval.merge_metric' has no attribute 'z_transitions'`.

- [ ] **Step 3: Implement `z_transitions` and `summarize_z_consistency`**

In `eval/merge_metric.py`, add after `score_chain` (before `summarize`):

```python
def z_transitions(masks: dict[int, tuple[np.ndarray, int, int]]) -> list[dict]:
    """Per-transition z-to-z consistency records for one chain's raw masks.

    For each pair of masks adjacent in sorted z order (masks is {z: (mask, x0, y0)}
    from pipeline.chain_masks_in_sam), computes IoU and centroid drift in a shared
    coordinate frame built from their own _sam-grid offsets: two masks can have
    different local shapes (a tier-2 crop window can move or resize between frames),
    so this pastes both onto a canvas sized to their combined bounding box rather
    than comparing local arrays directly.

    A transition touching an empty mask (matching the empty = not mask.any()
    convention score_chain already uses) gets iou=None, centroid_drift_px=None:
    dropout is the existing empty/dropout_rate signal's job, not a consistency
    reading, so it is not conflated with a low-IoU score here."""
    zs = sorted(masks.keys())
    out: list[dict] = []
    for z_from, z_to in zip(zs, zs[1:]):
        mask_a, x0_a, y0_a = masks[z_from]
        mask_b, x0_b, y0_b = masks[z_to]
        rec = {"z_from": int(z_from), "z_to": int(z_to), "gap": int(z_to - z_from),
               "iou": None, "centroid_drift_px": None}
        if not mask_a.any() or not mask_b.any():
            out.append(rec)
            continue
        h_a, w_a = mask_a.shape[:2]
        h_b, w_b = mask_b.shape[:2]
        x_min = min(x0_a, x0_b)
        y_min = min(y0_a, y0_b)
        x_max = max(x0_a + w_a, x0_b + w_b)
        y_max = max(y0_a + h_a, y0_b + h_b)
        canvas_a = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_b = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_a[y0_a - y_min:y0_a - y_min + h_a, x0_a - x_min:x0_a - x_min + w_a] = mask_a
        canvas_b[y0_b - y_min:y0_b - y_min + h_b, x0_b - x_min:x0_b - x_min + w_b] = mask_b
        intersection = int((canvas_a & canvas_b).sum())
        union = int((canvas_a | canvas_b).sum())
        rec["iou"] = intersection / union if union > 0 else 0.0

        ys_a, xs_a = np.where(mask_a)
        ys_b, xs_b = np.where(mask_b)
        cx_a, cy_a = float(xs_a.mean()) + x0_a, float(ys_a.mean()) + y0_a
        cx_b, cy_b = float(xs_b.mean()) + x0_b, float(ys_b.mean()) + y0_b
        rec["centroid_drift_px"] = float(np.hypot(cx_b - cx_a, cy_b - cy_a))
        out.append(rec)
    return out


def summarize_z_consistency(transitions: list[dict], *, low_iou_threshold: float = 0.5) -> dict:
    """Aggregate z_transitions records (one chain's, or a whole run's concatenated).

    n_dropout_transitions counts transitions with iou=None separately from the
    IoU/drift means, so a chain that scores well only because most transitions were
    skipped by dropout does not look falsely consistent. frac_low_iou is restricted
    to gap==1 transitions (a gap>1 transition spans real missed frames, not a
    consistency failure, and would understate the run's true low-IoU rate if mixed
    in)."""
    n = len(transitions)
    dropout = [t for t in transitions if t["iou"] is None]
    scored = [t for t in transitions if t["iou"] is not None]
    gap1_scored = [t for t in scored if t["gap"] == 1]
    return {
        "n_transitions": n,
        "n_dropout_transitions": len(dropout),
        "mean_z2z_iou": float(np.mean([t["iou"] for t in scored])) if scored else None,
        "mean_centroid_drift_px": (
            float(np.mean([t["centroid_drift_px"] for t in scored])) if scored else None),
        "frac_gap1_transitions": (
            sum(1 for t in transitions if t["gap"] == 1) / n) if n else None,
        "frac_low_iou": (
            sum(1 for t in gap1_scored if t["iou"] < low_iou_threshold) / len(gap1_scored)
        ) if gap1_scored else None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_z_consistency.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check eval/merge_metric.py tests/test_z_consistency.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass (no regressions in the existing suite).

- [ ] **Step 6: Commit**

```bash
git add eval/merge_metric.py tests/test_z_consistency.py
git commit -m "feat(merge_metric): add z_transitions and summarize_z_consistency"
```

---

### Task 2: wire into `score_run`, `format_summary`, and the CLI

**Files:**
- Modify: `eval/merge_metric.py`

**Interfaces:**
- Consumes: `z_transitions`, `summarize_z_consistency` from Task 1 (exact signatures above).
- Produces: `score_run(...)` now also writes `<root>/_z_consistency.csv` and its returned summary
  dict carries the new keys (`n_transitions`, `n_dropout_transitions`, `mean_z2z_iou`,
  `mean_centroid_drift_px`, `frac_gap1_transitions`, `frac_low_iou`) alongside the existing ones.
  `score_run` gains a `low_iou_threshold: float = 0.5` keyword parameter, and `main()` gains a
  `--low-iou-threshold` CLI flag that threads through to it. No other task in this plan depends on
  these, this is the last code task.

- [ ] **Step 1: Modify `score_run`**

In `eval/merge_metric.py`, `score_run`'s signature currently is:

```python
def score_run(root, annotate_df: pd.DataFrame | None = None,
              radius: int = DEFAULT_RADIUS, membrane_source="auto",
              tau: float = membrane.DEFAULT_TAU, tol: int = membrane.DEFAULT_TOL,
              scale: int | None = None, neurons=None, out_csv=None
              ) -> tuple[pd.DataFrame, dict]:
```

Add `low_iou_threshold: float = 0.5` as a new keyword parameter (after `out_csv`). Inside the
function, the existing per-chain loop is:

```python
    rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        if want is not None and neuron not in want:
            continue
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius,
                                   membrane_source, tau=tau, tol=tol):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)

    per = pd.DataFrame(rows)
    summary = summarize(per)
    if len(per):
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        dest = Path(out_csv) if out_csv is not None else root / "_merge_metric.csv"
        per_out.to_csv(dest, index=False)
    return per, summary
```

Replace it with (the per-frame loop body is unchanged, only the z-consistency pass is new, added
inside the same chain loop so `chain_dir` is in scope, and the new CSV write/summary fold happens
after):

```python
    rows: list[dict] = []
    z_rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        if want is not None and neuron not in want:
            continue
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius,
                                   membrane_source, tau=tau, tol=tol):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)
            chain_masks = pipeline.chain_masks_in_sam(chain_dir)
            for trec in z_transitions(chain_masks):
                trec.update(neuron=neuron, chain_idx=cidx)
                z_rows.append(trec)

    per = pd.DataFrame(rows)
    summary = summarize(per)
    z_summary = summarize_z_consistency(z_rows, low_iou_threshold=low_iou_threshold)
    summary.update(z_summary)
    if len(per):
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        dest = Path(out_csv) if out_csv is not None else root / "_merge_metric.csv"
        per_out.to_csv(dest, index=False)
    if z_rows:
        z_dest = root / "_z_consistency.csv"
        pd.DataFrame(z_rows).to_csv(z_dest, index=False)
    return per, summary
```

Note: `z_rows` always gets a chain's transitions even when that chain contributed zero rows to
`rows` (e.g. every frame's `score_chain` record was filtered out for some future reason); this
plan does not special-case that, `z_transitions` and `score_chain` are independent passes over the
same `chain_masks_in_sam` output, matching the spec.

- [ ] **Step 2: Update `format_summary`**

Current:

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

Add a third segment after the membrane one:

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
    if s.get("mean_z2z_iou") is not None:
        line += (f" | mean_z2z_iou={s['mean_z2z_iou']:.3f} "
                 f"mean_centroid_drift_px={s['mean_centroid_drift_px']:.2f} "
                 f"frac_low_iou={s['frac_low_iou']:.3f} "
                 f"frac_gap1={s['frac_gap1_transitions']:.3f}")
    return line
```

- [ ] **Step 3: Add the CLI flag**

In `main()`, after the existing `--tol` argument:

```python
    ap.add_argument("--low-iou-threshold", type=float, default=0.5,
                    help="z-to-z IoU below this counts toward frac_low_iou")
```

Find where `score_run` is actually called inside `main()` (search for `score_run(` in the function)
and add `low_iou_threshold=args.low_iou_threshold` to that call's keyword arguments.

- [ ] **Step 4: Run the full test suite**

Run: `py -3 -m pytest -q`
Expected: all tests pass, including the existing `tests/test_merge_metric.py` (confirms
`score_chain`'s untouched signature still works and `score_run`'s new pass does not break the
existing per-frame scoring path).
Run: `ruff check eval/merge_metric.py`
Expected: clean.

- [ ] **Step 5: Smoke-test against a real tree**

F: drive must be mounted (`ls "F:/ZhenLab/Data/output_masks/resolution_experiments/"` should list
directories; if not, stop and report BLOCKED rather than guessing at output). Run:

```bash
py -3 -m eval.merge_metric --root "F:/ZhenLab/Data/output_masks/resolution_experiments/original_perslice_only_guard_merged" --no-membrane
```

Expected: runs to completion, prints a summary line containing the new
`mean_z2z_iou=... mean_centroid_drift_px=... frac_low_iou=... frac_gap1=...` segment, and writes
`F:/ZhenLab/Data/output_masks/resolution_experiments/original_perslice_only_guard_merged/_z_consistency.csv`.
Report the actual printed summary line in your report, this is a real number worth recording, not
just a smoke check.

- [ ] **Step 6: Commit**

```bash
git add eval/merge_metric.py
git commit -m "feat(merge_metric): wire z-to-z consistency into score_run and the CLI"
```

---

### Task 3: docs

**Files:**
- Modify: `docs/reference/cli.md`
- Modify: `docs/explanation/roadmap.md`
- Modify: `docs/CHANGELOG.md`

**Interfaces:**
- Consumes: Task 2's real smoke-test output (the actual summary line and its numbers) for the docs
  to cite, if a concrete number is worth including; otherwise describe the fields without inventing
  numbers.

- [ ] **Step 1: Update `docs/reference/cli.md`**

Find the `eval.merge_metric` section (it documents `foreign_frame_rate`, `mild_bleed_rate`, etc.
around the area that already describes the summary line fields). Add the new fields:
`mean_z2z_iou`, `mean_centroid_drift_px`, `frac_gap1_transitions`, `frac_low_iou`,
`n_dropout_transitions` (each with a one-line description of what it means, matching the existing
entries' terseness), and the new `--low-iou-threshold` flag (default 0.5).

- [ ] **Step 2: Update the roadmap**

In `docs/explanation/roadmap.md`, find item 12 in section 5b (search for "z-to-z consistency
metric"). Change its status from "TODO" to "DONE 2026-07-29", briefly describing what was built
(`z_transitions`/`summarize_z_consistency` in `eval/merge_metric.py`, a new `_z_consistency.csv`
per run tree) and noting explicitly that retro-scoring specific trees for a per-slice-vs-propagation
comparison is a follow-on use of the tool, not something this landing did (per the spec's stated
scope). Also check section 5 (search for "`0.a`") and update that bullet the same way if it also
carries a TODO/pending marker.

- [ ] **Step 3: Add a CHANGELOG entry**

Follow the existing entries' format (anchor id, `##` heading, prose, `---` separator,
most-recent-first in both the file body and the Contents list). Cover what was built and why
(the per-frame ruler's blindness to temporal consistency, and what this adds).

- [ ] **Step 4: Humanize and check for em dashes**

Run the `humanizer` skill on everything written in Steps 1-3 before committing (required by this
project's CLAUDE.md, not optional). Then:

Run: `grep -n $'\xe2\x80\x94\|\xe2\x80\x93' docs/reference/cli.md docs/explanation/roadmap.md docs/CHANGELOG.md`
Expected: no output.

- [ ] **Step 5: Run the full suite one last time**

Run: `py -3 -m pytest -q`
Expected: all tests pass (this task shouldn't touch code, but confirm nothing drifted).

- [ ] **Step 6: Commit**

```bash
git add docs/reference/cli.md docs/explanation/roadmap.md docs/CHANGELOG.md
git commit -m "docs(roadmap): record the z-to-z consistency metric landing"
```
