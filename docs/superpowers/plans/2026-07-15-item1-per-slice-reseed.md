# Item 1: per-slice re-seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Segment each slice of a chain independently, re-seeded from that slice's own skeleton node inside the chain's shared tier-2 crop, with no video propagation, so SAM2 memory can never carry the wrong cell across slices.

**Architecture:** Reuse the existing tier-2 chain crop and prepared `_pcrop` frames. Replace only the `propagate()` call in `run_chain` (behind a config flag) with a per-slice loop that runs image-mode on each frame seeded from that frame's node, returning the SAME `(video_segments, frame_conf, pred_iou)` structure `propagate()` returns, so all downstream save/QC is unchanged. Nodes are ~one-per-slice already (virtual nodes fill gaps); a residual gap is linearly interpolated.

**Tech Stack:** Python, numpy, OpenCV, SAM2 image predictor (GPU for real runs), pytest (CPU-only, torch-free via a stub predictor).

## Global Constraints

- No em dashes anywhere in code, comments, or commit messages.
- Tests are CPU-only and torch-free (`py -3 -m pytest`); use a stub image-predictor for the loop test.
- `ruff check .` clean; touch only the files each task names.
- Import direction: library (`pipeline`, `sam2_utils`) never imports drivers or `eval`.
- Default behavior byte-identical when `per_slice_reseed` is False: the new branch is only entered when the flag is on.
- Per-slice masks must be returned in the SAME space and structure as `propagate()` (`{frame_idx: {obj_id: mask_bool}}` in the chain's propagation space) so save/QC/`chain_masks_in_sam` need no change.
- Local smoke ALWAYS downscaled (small model, high scale/crop_scale, few short chains); full-res only on CCDB.

---

### Task 1: Centreline point per slice (`centreline_by_z`)

**Files:**
- Modify: `pipeline/predict.py` (add near `build_prompts`)
- Test: `tests/test_per_slice_reseed.py`

**Interfaces:**
- Produces: `centreline_by_z(chain: dict, annotate_df: pandas.DataFrame) -> dict[int, tuple[float, float]]` mapping each catmaid_z in the chain's node z-range to a `(x_tif, y_tif)` centreline point. Uses every node of the chain (real and virtual). For a z inside the range with no node, linearly interpolate between the nearest lower and higher present z.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_per_slice_reseed.py
import pandas as pd
from pipeline import predict

def test_centreline_by_z_uses_nodes_and_interpolates_gaps():
    # chain nodes at z=1400 (x_tif=80,y_tif=800) and z=1402 (x_tif=120,y_tif=840); z=1401 is a gap
    chain = {"cell_name": "AVAL", "nodes": ["n0", "n2"]}
    df = pd.DataFrame({
        "node_id": ["n0", "n2", "other"],
        "cell_name": ["AVAL", "AVAL", "AVBR"],
        "z": [1400, 1402, 1401],
        "x_tif": [80.0, 120.0, 9999.0],   # 'other' is a different neuron, must be ignored
        "y_tif": [800.0, 840.0, 9999.0],
    })
    got = predict.centreline_by_z(chain, df)
    assert set(got) == {1400, 1401, 1402}
    assert got[1400] == (80.0, 800.0)
    assert got[1402] == (120.0, 840.0)
    # z=1401 interpolated halfway between the two chain nodes, NOT the foreign node
    assert got[1401] == (100.0, 820.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_centreline_by_z_uses_nodes_and_interpolates_gaps -v`
Expected: FAIL, `AttributeError: module 'pipeline.predict' has no attribute 'centreline_by_z'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/predict.py (add near build_prompts)
def centreline_by_z(chain: dict, annotate_df: "pd.DataFrame") -> dict:
    """{catmaid_z: (x_tif, y_tif)} centreline point for each z in the chain's node
    z-range. Uses the chain's own nodes (real and virtual); a z with no node is
    linearly interpolated between the nearest present z's (which is what a virtual
    node already is). Foreign neurons' nodes are never used."""
    ids = {str(n) for n in chain["nodes"]}
    sub = annotate_df[annotate_df["node_id"].astype(str).isin(ids)]
    by_z: dict[int, tuple[float, float]] = {}
    for z, x, y in zip(sub["z"].astype(int), sub["x_tif"].astype(float), sub["y_tif"].astype(float)):
        by_z.setdefault(int(z), (float(x), float(y)))   # first node wins on a shared z
    if not by_z:
        return {}
    present = sorted(by_z)
    out: dict[int, tuple[float, float]] = {}
    for z in range(present[0], present[-1] + 1):
        if z in by_z:
            out[z] = by_z[z]
            continue
        lo = max(p for p in present if p < z)
        hi = min(p for p in present if p > z)
        t = (z - lo) / (hi - lo)
        (x0, y0), (x1, y1) = by_z[lo], by_z[hi]
        out[z] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_centreline_by_z_uses_nodes_and_interpolates_gaps -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/predict.py tests/test_per_slice_reseed.py
git commit -m "feat(pipeline): per-slice centreline point with vnode/gap fill"
```

---

### Task 2: `per_slice_reseed` config flag

**Files:**
- Modify: `pipeline/config.py`
- Test: `tests/test_per_slice_reseed.py`

**Interfaces:**
- Produces: `PipelineConfig.per_slice_reseed: bool = False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_per_slice_reseed.py (append)
from pipeline import PipelineConfig

def test_per_slice_reseed_flag_defaults_false():
    assert PipelineConfig().per_slice_reseed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_per_slice_reseed_flag_defaults_false -v`
Expected: FAIL, `AttributeError` / `TypeError` on the missing field.

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/config.py` `PipelineConfig` (near the other propagation/seed knobs), matching the file's dataclass style:

```python
    # Per-slice re-seeding (roadmap Phase 1 item 1). When True, run_chain segments each
    # slice independently in the chain crop, re-seeded from that slice's own node, instead
    # of seeding one anchor and propagating. Memory cannot carry the wrong cell across
    # slices. Default False keeps the propagation path byte-identical.
    per_slice_reseed: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_per_slice_reseed_flag_defaults_false -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_per_slice_reseed.py
git commit -m "feat(config): add per_slice_reseed flag (default off)"
```

---

### Task 3: Per-slice segmentation over prepared frames (`segment_per_slice`)

**Files:**
- Modify: `pipeline/propagate.py` (add alongside the module-level `propagate`)
- Test: `tests/test_per_slice_reseed.py`

**Context for the implementer:** Read `propagate()` in `pipeline/propagate.py` and `image_predict` + `anchor_crop_predict` in `pipeline/predict.py` before starting. `propagate()` returns `(video_segments, frame_conf, pred_iou)` where `video_segments` is `{frame_idx: {obj_id: mask_bool}}`. Your function returns the SAME shape. The frames for a chain are prepared on disk (one image per frame_idx) exactly as the propagation path consumes them; read them with the same helper `propagate`/`PropagationSession` uses (inspect how `PropagationSession` loads a frame, and reuse that path). The node for each frame comes from `centreline_by_z` (Task 1) mapped into the frame's prediction space; `frame_to_z` (available in `run_chain`, passed in) maps frame_idx to catmaid_z.

**Interfaces:**
- Produces: `segment_per_slice(image_predictor, frames_dir, frame_to_z, centreline_tif, annotate_df, *, cfg, obj_id, cw=None) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]`. For each frame_idx: load the frame image, map that frame's centreline point (and neighbour negatives from `build_prompts`) into the prediction space, run `image_predict` (single mask or multimask per `cfg.multimask_anchor`), and collect the mask. Returns `(video_segments, frame_conf, pred_iou)` matching `propagate()`. `cw` is the chain crop window (the prediction space is `_pcrop` when `cw` is set, `_sam` when None); reuse the same coordinate mapping `anchor_crop_predict` uses to place points in that space.

- [ ] **Step 1: Write the failing test (CPU, stub predictor)**

```python
# tests/test_per_slice_reseed.py (append)
import numpy as np
from pipeline import propagate as prop
from pipeline import config as cfgmod

class _StubPredictor:
    """Minimal image-predictor stub: set_image records shape, predict returns one
    canned mask covering the seeded point, so segment_per_slice runs torch-free."""
    def set_image(self, img): self._hw = img.shape[:2]
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        h, w = self._hw
        m = np.zeros((h, w), dtype=bool)
        if point_coords is not None and len(point_coords):
            x, y = int(point_coords[0][0]), int(point_coords[0][1])
            m[max(0, y-2):y+3, max(0, x-2):x+3] = True
        masks = np.stack([m, m, m]) if multimask_output else m[None]
        scores = np.array([0.9, 0.8, 0.7][: masks.shape[0]])
        logits = np.zeros((masks.shape[0], h, w), dtype=np.float32)
        return masks, scores, logits

def test_segment_per_slice_returns_a_mask_per_frame(tmp_path):
    # 3 frames on disk, full-frame (cw=None -> _sam space)
    import cv2
    fdir = tmp_path / "frames"; fdir.mkdir()
    for i in range(3):
        cv2.imwrite(str(fdir / f"{i:05d}.jpg"), np.full((40, 40, 3), 127, np.uint8))
    frame_to_z = {0: 1400, 1: 1401, 2: 1402}
    centreline_tif = {1400: (80.0, 80.0), 1401: (80.0, 80.0), 1402: (80.0, 80.0)}  # scale 8 -> (10,10)
    import pandas as pd
    df = pd.DataFrame({"node_id": [], "cell_name": [], "z": [], "x_tif": [], "y_tif": []})
    cfg = cfgmod.PipelineConfig(scale=8, k_max_neg=0)
    vs, conf, piou = prop.segment_per_slice(
        _StubPredictor(), str(fdir), frame_to_z, centreline_tif, df,
        cfg=cfg, obj_id=1, cw=None)
    assert set(vs) == {0, 1, 2}
    assert all(1 in vs[f] for f in vs)              # obj_id present per frame
    assert vs[0][1].sum() > 0                        # canned mask non-empty at the seed
    assert set(conf) == {0, 1, 2} and set(piou) == {0, 1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_segment_per_slice_returns_a_mask_per_frame -v`
Expected: FAIL, `AttributeError: module 'pipeline.propagate' has no attribute 'segment_per_slice'`.

- [ ] **Step 3: Write minimal implementation**

Implement `segment_per_slice` per the Interfaces contract. Requirements the code must meet (the implementer writes the body after reading `image_predict` / `anchor_crop_predict` / how `PropagationSession` loads frames):
- Iterate `frame_to_z` in frame order; load each frame image with the same reader the propagation path uses.
- For each frame, build a positive point from `centreline_tif[z]` mapped into the prediction space (`_sam`: divide tif by `cfg.scale`; `_pcrop`: `_sam` then `cw.tif_to_crop`, reusing `anchor_crop_predict`'s mapping), plus neighbour negatives from `build_prompts` when `cfg.k_max_neg > 0`.
- Call `image_predict(image_predictor, image, prompts, multimask=cfg.multimask_anchor, ...)`; take its mask.
- Collect `video_segments[frame_idx] = {obj_id: mask_bool}`, and populate `frame_conf` / `pred_iou` per frame (use the returned score for `pred_iou`; a mean-foreground proxy for `frame_conf`, matching `propagate`'s semantics).
- No `torch` import at module top that would break the CPU test; the predictor is injected.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py::test_segment_per_slice_returns_a_mask_per_frame -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/propagate.py tests/test_per_slice_reseed.py
git commit -m "feat(pipeline): segment_per_slice, node-anchored per-frame image-mode"
```

---

### Task 4: Wire `per_slice_reseed` into `run_chain`

**Files:**
- Modify: `pipeline/orchestrator.py`
- Test: `tests/test_per_slice_reseed.py`

**Context for the implementer:** Read `run_chain` in `pipeline/orchestrator.py`, specifically the `propagate(...)` call (around line 351) that yields `video_segments, frame_conf, pred_iou`, and how `frame_to_z`, the chain crop window, and `annotate_df` are already in scope there. Add a branch: when `cfg.per_slice_reseed`, compute `centreline = predict.centreline_by_z(chain, annotate_df)` and call `propagate.segment_per_slice(...)` to produce the same three return values, instead of `propagate(...)`. Everything after (save, QC, state) is unchanged.

**Interfaces:**
- Consumes: Task 1 `centreline_by_z`, Task 3 `segment_per_slice`, `cfg.per_slice_reseed`.
- Produces: no new public symbol; a behavior branch in `run_chain`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_per_slice_reseed.py (append)
def test_run_chain_routes_to_per_slice(monkeypatch):
    # verify the flag routes run_chain through segment_per_slice, not propagate
    import pipeline.orchestrator as orch
    calls = {"per_slice": 0, "propagate": 0}
    monkeypatch.setattr(orch, "segment_per_slice",
                        lambda *a, **k: ({0: {1: __import__("numpy").zeros((4, 4), bool)}}, {0: 0.0}, {0: 0.0}),
                        raising=False)
    monkeypatch.setattr(orch, "propagate",
                        lambda *a, **k: ({}, {}, {}), raising=False)
    # The implementer exposes a thin helper `_do_segmentation(state, cfg, ...)` that run_chain
    # calls, returning (video_segments, frame_conf, pred_iou); this test calls it directly with
    # per_slice_reseed True and asserts segment_per_slice was used. See Step 3.
    # (Concrete assertion finalized by the implementer against the extracted helper.)
    assert hasattr(orch, "segment_per_slice") or True
```

Note to implementer: replace the placeholder assertion above with a real one against a small extracted helper. Extract the segmentation dispatch (the `if cfg.per_slice_reseed: ... else: propagate(...)` choice) into a module-level `_do_segmentation(...)` in `orchestrator.py` that returns `(video_segments, frame_conf, pred_iou)`, so it can be unit-tested with both flag values using stubs, without standing up all of `run_chain`. The test must assert the flag selects the per-slice path.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py -k route -v`
Expected: FAIL until `_do_segmentation` exists and routes on the flag.

- [ ] **Step 3: Write minimal implementation**

Extract `_do_segmentation(cfg, *, video_predictor, image_predictor, frames_dir, frame_to_z, prompts, anchor_frame_idx, chain, annotate_df, cw, obj_id, ...) -> (video_segments, frame_conf, pred_iou)` in `orchestrator.py`. Body:

```python
def _do_segmentation(cfg, *, image_predictor, video_predictor, frames_dir, frame_to_z,
                     prompts, anchor_frame_idx, chain, annotate_df, cw, obj_id, **seed_kw):
    if cfg.per_slice_reseed:
        centreline = predict.centreline_by_z(chain, annotate_df)
        return segment_per_slice(image_predictor, frames_dir, frame_to_z, centreline,
                                 annotate_df, cfg=cfg, obj_id=obj_id, cw=cw)
    return propagate(video_predictor, frames_dir, prompts, anchor_frame_idx,
                     obj_id=obj_id, **seed_kw)
```

Replace the direct `propagate(...)` call in `run_chain` with a call to `_do_segmentation(...)`, passing the arguments already in scope. Finalize the Step 1 test to monkeypatch `segment_per_slice` and `propagate` and assert `_do_segmentation` with `per_slice_reseed=True` calls the former and `False` calls the latter.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_per_slice_reseed.py -v`
Expected: PASS (all per-slice tests), and `py -3 -m pytest -q` shows no regressions.

- [ ] **Step 5: Commit**

```bash
git add pipeline/orchestrator.py tests/test_per_slice_reseed.py
git commit -m "feat(pipeline): route run_chain through per-slice re-seed when flagged"
```

---

### Task 5: Preset + downscaled local smoke (human-run) + CCDB note

**Files:**
- Modify: `sam2_utils/presets.py`
- Modify: `docs/how-to/run-on-narval.md` (add a per-slice line)

**Interfaces:**
- Produces: a `original_perslice` preset mirroring `original_tier2_s1forced_neg` plus `per_slice_reseed=True`, on `EXP_NEURONS`.

- [ ] **Step 1: Add the preset**

Add to `PRESETS` in `sam2_utils/presets.py`, mirroring `original_tier2_s1forced_neg`'s pipeline knobs, adding `"per_slice_reseed": True`, output_root `config.OUTPUT_ROOT.parent / "exp_perslice"`, neurons `EXP_NEURONS`.

- [ ] **Step 2: Verify the preset builds a valid config**

Run:
```bash
py -3 -c "from sam2_utils import presets; from pipeline import PipelineConfig; p=presets.get_preset('original_perslice'); print(PipelineConfig(**p['pipeline']).per_slice_reseed)"
```
Expected: prints `True`.

- [ ] **Step 3: Document the downscaled local smoke and the CCDB run**

In `docs/how-to/run-on-narval.md`, under the resolution-experiments section, add a short block: the DOWNSCALED local smoke first, then the Narval submit line.

Local smoke (downscaled, a few short chains, on the local GPU):
```bash
py -3 batch.py --preset original_perslice --neurons AIYL --model-size tiny --clean
py -3 -m eval.merge_metric --root <that run's output_root>
```
(small model + the preset's coarse scale keep it light; never run full-res locally.)

Narval (full run):
```bash
sbatch --job-name=exp_perslice --export=ALL,EXP_PRESET=original_perslice cluster/run_exp.sh
```

- [ ] **Step 4: Commit**

```bash
git add sam2_utils/presets.py docs/how-to/run-on-narval.md
git commit -m "feat(presets): original_perslice preset + downscaled smoke / CCDB docs"
```

- [ ] **Step 5: HUMAN-RUN GPU smoke (notify, do not attempt to fake)**

The real verification is a GPU run: it needs the SAM2 checkpoint, the worm tifs, and a GPU, so it is run by the human on their machine (downscaled) and then on CCDB. The controller must NOT mark item 1 "done" on unit tests alone; notify the human with the Step 3 commands and the A/B target (per-slice vs `tier2_s1forced_neg` on `eval.merge_metric`: foreign_frame_rate down, dropout not up).

---

## Notes for the implementer
- The whole point is that per-slice output plugs into the unchanged save/QC path by matching `propagate()`'s return contract. If you find yourself changing save/QC or `chain_masks_in_sam`, stop and reconsider: the masks should live in the same space (`_pcrop` with the chain `cw`, or `_sam` when `cw` is None) as the propagation path already produces.
- Reuse `build_prompts` for neighbour negatives and the exact `_sam`->`_pcrop` point mapping from `anchor_crop_predict`; do not reinvent coordinate math.
- Generous-capped multimask selection (item 3) is a SEPARATE plan; here just forward `cfg.multimask_anchor` to `image_predict` as the propagation path already does. Item 3 plugs in later through the same `image_predict` call.
