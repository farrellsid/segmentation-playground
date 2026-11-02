# Per-frame segmentation, Plan 2: Approach 2 (auto-mask) + AMG tuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. This plan builds on Plan 1 (`2026-07-20-perframe-foundation-and-approach1.md`), which must be landed first: it provides `sam2_utils.perframe` (F1, F3 resolvers, `select_by_metric`, `match_amg_to_nodes`), `eval.perframe_score.score_frame`, and `run_perframe.py`.

**Goal:** Add Approach 2 (SAM2 automatic-mask generation, match masks to nodes, keep the rest as competitors, resolve overlaps membrane-aware) and an AMG parameter tuner that searches the AMG knobs against our per-frame metric.

**Architecture:** Approach 2 is a second mode in `run_perframe.py` reusing Plan 1's matcher/resolvers/scorer. The tuner is a grid/random search over the AMG knobs scored by the F2 composite on the frame sample, with mandatory visual dumps. Design: `docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md`.

**Tech Stack:** Python, numpy, SAM2 `SAM2AutomaticMaskGenerator`, pytest.

## Global Constraints

- No em dashes anywhere.
- Tests CPU-only and torch-free; AMG-touching code gets a fake-AMG CPU smoke, real runs are GPU/CCDB.
- Lint clean on touched files; library never imports drivers or `eval`.
- Every run keeps `results/perframe/<run>/` and appends to `docs/explanation/perframe-experiments.md`.
- F2 composite (tuner objective): require own-node containment and zero foreign, then maximise `boundary_on_membrane`, then minimise `spanning_rate` and `overlap_fraction`.

---

### Task 1: Approach 2 runner (auto-mask + match + keep competitors)

**Files:**
- Modify: `run_perframe.py` (add `segment_frame_amg` + `--approach amg`)
- Test: `tests/test_run_perframe_amg_smoke.py`

**Interfaces:**
- Consumes: `sam2_utils.perframe.match_amg_to_nodes`, `resolve_overlaps_argmax` / `resolve_overlaps_watershed`; `eval.perframe_score.score_frame`; a thin `build_amg(sam2_model, params) -> SAM2AutomaticMaskGenerator` wrapper.
- Produces: `segment_frame_amg(amg, frame_sam, node_index, membrane_map, *, match, resolver, cfg) -> (cell_masks, label_map, score)`; `build_amg(model, **amg_params)`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_run_perframe_amg_smoke.py
import numpy as np
import run_perframe


class FakeAMG:
    """Stand-in for SAM2AutomaticMaskGenerator.generate: returns AMG-style dicts."""
    def __init__(self, masks): self._masks = masks
    def generate(self, image):
        return [{"segmentation": m, "area": int(m.sum()), "predicted_iou": 0.9,
                 "stability_score": 0.95} for m in self._masks]


def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_segment_frame_amg_labels_and_keeps_competitors():
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    amg = FakeAMG([_disk(10, 10, 5), _disk(30, 30, 5), _disk(20, 4, 3)])  # 2 cells + 1 junk
    cell_masks, lab, score = run_perframe.segment_frame_amg(
        amg, frame, node_index, mem, match="metric", resolver="argmax",
        cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_run_perframe_amg_smoke.py -v`
Expected: FAIL with `AttributeError: module 'run_perframe' has no attribute 'segment_frame_amg'`.

- [ ] **Step 3: Implement `build_amg` + `segment_frame_amg`**

Add to `run_perframe.py`:

```python
def build_amg(sam2_model, **amg_params):
    """Thin wrapper: SAM2AutomaticMaskGenerator(sam2_model, **amg_params). Kept here (driver)
    so the library stays torch-free. amg_params: points_per_side, pred_iou_thresh,
    stability_score_thresh, stability_score_offset, box_nms_thresh, crop_n_layers,
    crop_n_points_downscale_factor, min_mask_region_area, use_m2m."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(model=sam2_model, **amg_params)


def segment_frame_amg(amg, frame_sam, node_index, membrane_map, *,
                      match: str, resolver: str, cfg):
    """One frame, AMG mode. amg.generate(frame) -> masks; match each node to its mask and
    keep the rest as competitors (pf.match_amg_to_nodes when match=='metric'; a smallest-
    containing-mask rule when match=='area'); resolve overlaps over targets + competitors so
    competitors push back, then keep only the target labels; score the labelled subset.

    - anns = amg.generate(frame_sam); amg_masks = [a['segmentation'].astype(bool) for a in anns].
    - labels, leftover = pf.match_amg_to_nodes(amg_masks, node_index, membrane_map) for 'metric';
      for 'area' pick, per node, the smallest amg mask containing it (pipeline._point_in_mask),
      leftover = the unmatched.
    - resolution: order = labelled cells (seed = their node xy) then competitors (seed =
      centroid of the competitor mask); run pf.resolve_overlaps_{argmax,watershed}. Then map
      the label_map back to cell names, dropping competitor labels to background.
    - cell_masks[cell] = (label_map == cell_index). score = score_frame(cell_masks, node_index,
      membrane_map, radius=cfg.radius, tau=cfg.tau). Return (cell_masks, label_map_cellonly, score).
    """
    # ... (per the notes; the FakeAMG smoke exercises the control flow)
```

Then wire `--approach amg` in `_run` alongside `prompt`, with `--match area|metric` and the shared `--resolver`. AMG params come from `--amg-params <json>` or defaults matching the notebook's `mask_generator_2`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_run_perframe_amg_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: CPU smoke on a real downscaled frame**

Run: `py -3 run_perframe.py --approach amg --frames 1400 --match metric --resolver argmax --scale 8 --model-size tiny --out results/perframe/amg_smoke`
Expected: writes the run dir + montage; prints scores. Eyeball the montage (AMG on EM is the risky part).

- [ ] **Step 6: Commit**

```bash
git add run_perframe.py tests/test_run_perframe_amg_smoke.py
git commit -m "feat(run_perframe): Approach 2 auto-mask + match + keep-competitors"
```

---

### Task 2: AMG parameter tuner

**Files:**
- Modify: `eval/perframe_score.py` (add `objective(score) -> float`)
- Modify: `run_perframe.py` (add `--tune` mode)
- Test: `tests/test_perframe_objective.py`

**Interfaces:**
- Consumes: `segment_frame_amg`, `score_frame`.
- Produces: `eval.perframe_score.objective(score: dict) -> float` (the scalar the tuner maximises); a `--tune` mode that searches AMG params over the frame sample.

- [ ] **Step 1: Write the failing test for the objective**

```python
# tests/test_perframe_objective.py
from eval.perframe_score import objective


def test_objective_rewards_coverage_penalises_bleed_and_overlap():
    good = {"own_coverage": 1.0, "foreign_frame_rate": 0.0, "total_foreign": 0,
            "overlap_fraction": 0.0, "mean_boundary_on_membrane": 0.9, "spanning_rate": 0.0}
    bleed = {**good, "total_foreign": 20, "foreign_frame_rate": 0.5, "spanning_rate": 0.4}
    undercover = {**good, "own_coverage": 0.3}
    assert objective(good) > objective(bleed)
    assert objective(good) > objective(undercover)
    # None membrane fields must not crash
    nomem = {**good, "mean_boundary_on_membrane": None, "spanning_rate": None}
    assert isinstance(objective(nomem), float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_perframe_objective.py -v`
Expected: FAIL with `ImportError: cannot import name 'objective'`.

- [ ] **Step 3: Implement `objective` + the `--tune` loop**

Add to `eval/perframe_score.py`:

```python
def objective(score: dict) -> float:
    """Scalar the AMG tuner maximises, from a score_frame dict. Rewards own-node coverage
    and boundary-on-membrane; penalises foreign bleed, spanning, and pre-resolution overlap.
    Membrane terms are dropped (treated as 0 contribution) when None so a no-membrane run
    still ranks by coverage/bleed. Weights are a starting point, tune-able."""
    bo = score.get("mean_boundary_on_membrane") or 0.0
    sp = score.get("spanning_rate") or 0.0
    return (1.0 * score["own_coverage"]
            + 0.5 * bo
            - 1.0 * score["foreign_frame_rate"]
            - 0.5 * sp
            - 0.5 * score["overlap_fraction"])
```

Add `--tune` to `run_perframe.py`: iterate an AMG param grid (default a small grid over `pred_iou_thresh` in {0.7, 0.8, 0.88}, `stability_score_thresh` in {0.9, 0.95}, `points_per_side` in {32, 64}), for each build the AMG, run `segment_frame_amg` over `--frames`, average `objective` across frames, keep the best. Write every trial's `{params, mean_objective, per-frame scores}` to `results/perframe/<run>/trials.csv`, dump the best set's montages, and append the winner to the experiments log. Log a NOTE that the objective can be gamed and the montages are the real check.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_perframe_objective.py -v`
Expected: PASS.

- [ ] **Step 5: CPU smoke the tuner (tiny grid, one frame)**

Run: `py -3 run_perframe.py --tune --frames 1400 --scale 8 --model-size tiny --out results/perframe/tune_smoke`
Expected: `trials.csv` with one row per grid point, a best-set montage, a winner line in the experiments log.

- [ ] **Step 6: Commit**

```bash
git add eval/perframe_score.py run_perframe.py tests/test_perframe_objective.py
git commit -m "feat(perframe): AMG parameter tuner + composite objective"
```

---

### Task 3: Comparison run + docs

**Files:**
- Modify: `docs/explanation/perframe-experiments.md`, `docs/reference/cli.md`, `docs/CHANGELOG.md`, `docs/explanation/roadmap.md`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation + the A/B framing that Approach 1 vs Approach 2 (default) vs Approach 2 (tuned) is compared on the same frame sample.

- [ ] **Step 1: Document the comparison protocol**

In `docs/explanation/perframe-experiments.md`, add a "Comparison protocol" section: run Approach 1 (best knobs from Plan 1's sweep), Approach 2 default, and Approach 2 tuned on the same 5-to-10-frame sample, and read off `own_coverage` / `total_foreign` / `mean_boundary_on_membrane` / `overlap_fraction` plus the montages. State explicitly that the metric is incomplete and the montages are the deciding evidence.

- [ ] **Step 2: Update reference + history docs**

`docs/reference/cli.md`: the full `run_perframe.py` surface (`--approach prompt|amg`, `--negatives`, `--selection`, `--match`, `--resolver`, `--tune`, `--sweep`, `--frames`, `--scale`, `--model-size`, `--amg-params`, `--out`). `docs/CHANGELOG.md`: an entry for the per-frame thrust (both approaches + tuner), referencing the spec. `docs/explanation/roadmap.md`: note per-frame segmentation delivers Phase-2 2d (arbitration) and an early R5-lite probe.

- [ ] **Step 3: Humanize + verify**

Run the `humanizer` skill on the CHANGELOG / roadmap / experiments-log prose. Run `py -3 -m pytest -q && ruff check .`. Confirm no em dashes and links resolve.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: per-frame segmentation comparison protocol + CLI + CHANGELOG + roadmap"
```

---

## Self-review notes

Spec coverage: Approach 2 auto-mask + match + keep-competitors (T1), AMG tuner + composite objective (T2), comparison protocol + docs (T3). Reuses Plan 1's `match_amg_to_nodes`, resolvers, and `score_frame`. Competitors push back via being included (with centroid seeds) in the resolver, then dropped to background in the labelled output. The CCDB run of the full frame set (both approaches, tuned) is a launch step after both plans land, using the same `run_perframe.py` on a GPU node.
