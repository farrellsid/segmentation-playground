# SAM3 whole-set cluster evaluation Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--backend sam3` switch to `batch.py` so the whole target-worm set can be run with SAM3 on Narval and compared to the existing SAM2 baselines, with a local parity gate before any cluster job.

**Architecture:** One backend switch (a `build_predictors(cfg)` helper) routes to today's `setup.build_predictor` (sam2, default) or the already-built `sam2_utils.sam3_backend` adapters (sam3). Per-slice vs propagation stays a preset choice, so `--backend` is orthogonal. The cluster array and a Narval runbook carry it to scale; the run itself is human-executed (Duo-MFA).

**Tech Stack:** Python 3.13, existing `batch.py` / `pipeline` / `sam2_utils` / `eval`, the SAM3 adapters from the bake-off, Slurm (`cluster/`), HuggingFace `transformers` on Narval.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages.
- Run the `humanizer` skill on committed prose (the runbook, CHANGELOG, roadmap edits).
- No `Co-Authored-By: Claude` trailer.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. The backend-routing test must not load a model or import torch (monkeypatch the builders).
- `ruff check` clean on touched files only.
- The library (`pipeline.py`, `sam2_utils/`) must not import drivers or `eval`; `sam2_utils/sam3_backend.py` keeps torch/transformers lazy. `batch.py` is a driver and may import the adapters.
- SAM2 remains the default: `backend` defaults to `"sam2"`, and a run without `--backend sam3` must be byte-identical to today.
- SAM2 is NOT re-run in this phase; its baseline trees already exist and must never be overwritten (SAM3 writes to its own `--output-root`).
- Commit incrementally, one concern per commit.

---

## File structure

- Modify `pipeline/config.py`: add `backend: str = "sam2"` and `sam3_checkpoint: Optional[str] = None` to `PipelineConfig`.
- Modify `batch.py`: add `build_predictors(cfg)`; call it from `_build_session` and `_build_gt_session`; add `--backend` and `--sam3-checkpoint` CLI and thread them into the config.
- Create `tests/test_backend_selection.py`: CPU-only routing tests.
- Modify `cluster/run_array.sh` (+ its submit notes): forward backend / checkpoint / preset / output-root to `batch.py`.
- Create `docs/how-to/run-sam3-on-narval.md`: the deployment runbook.
- Modify `docs/CHANGELOG.md`, `docs/reference/code-map.md`, `docs/explanation/roadmap.md`: record the backend switch.

---

### Task 1: Backend switch (config + `build_predictors` + CLI)

**Files:**
- Modify: `pipeline/config.py` (add two fields to `PipelineConfig`)
- Modify: `batch.py` (`build_predictors`, both session builders, argparse)
- Test: `tests/test_backend_selection.py`

**Interfaces:**
- Produces: `batch.build_predictors(cfg: PipelineConfig) -> tuple[image_predictor, video_predictor]`. For `cfg.backend == "sam2"` returns `setup.build_predictor(...)` results (unchanged); for `"sam3"` returns `(Sam3ImagePredictor(ckpt), Sam3VideoPredictor(ckpt))` where `ckpt = cfg.sam3_checkpoint or sam3_backend.DEFAULT_CHECKPOINT_DIR`.
- `PipelineConfig` gains `backend: str = "sam2"`, `sam3_checkpoint: Optional[str] = None`.

- [ ] **Step 1: Add the config fields**

In `pipeline/config.py`, add to the `PipelineConfig` dataclass (near `model_size`):

```python
    backend: str = "sam2"                  # "sam2" (default, unchanged) or "sam3"
    sam3_checkpoint: Optional[str] = None  # SAM3 HF checkpoint dir; None -> adapter default
```

(`Optional` is already imported in this module; confirm and add to the import if not.)

- [ ] **Step 2: Write the failing routing tests**

Create `tests/test_backend_selection.py`:

```python
import batch
from pipeline import PipelineConfig


def test_build_predictors_sam2_uses_setup(monkeypatch):
    seen = {}
    def fake_build(**kw):
        seen[kw["kind"]] = kw
        return (f"SAM2_{kw['kind']}", None)
    monkeypatch.setattr(batch.setup, "build_predictor", fake_build)
    img, vid = batch.build_predictors(PipelineConfig(backend="sam2"))
    assert img == "SAM2_image" and vid == "SAM2_video"
    assert set(seen) == {"image", "video"}


def test_build_predictors_sam3_uses_adapters(monkeypatch):
    import sam2_utils.sam3_backend as sb
    monkeypatch.setattr(sb, "Sam3ImagePredictor", lambda ckpt: ("SAM3_image", ckpt))
    monkeypatch.setattr(sb, "Sam3VideoPredictor", lambda ckpt: ("SAM3_video", ckpt))
    img, vid = batch.build_predictors(PipelineConfig(backend="sam3", sam3_checkpoint="/x"))
    assert img == ("SAM3_image", "/x") and vid == ("SAM3_video", "/x")


def test_build_predictors_sam3_defaults_checkpoint(monkeypatch):
    import sam2_utils.sam3_backend as sb
    monkeypatch.setattr(sb, "Sam3ImagePredictor", lambda ckpt: ("i", ckpt))
    monkeypatch.setattr(sb, "Sam3VideoPredictor", lambda ckpt: ("v", ckpt))
    img, _ = batch.build_predictors(PipelineConfig(backend="sam3"))
    assert img[1] == sb.DEFAULT_CHECKPOINT_DIR
```

- [ ] **Step 3: Run to verify failure**

Run: `py -3 -m pytest tests/test_backend_selection.py -v`
Expected: FAIL with `AttributeError: module 'batch' has no attribute 'build_predictors'` (and/or `PipelineConfig` has no `backend`).

- [ ] **Step 4: Implement `build_predictors` and use it**

In `batch.py`, add near the session builders:

```python
def build_predictors(cfg: PipelineConfig):
    """Build (image_predictor, video_predictor) for the configured backend.

    sam2 (default): the existing setup.build_predictor path, byte-identical to before.
    sam3: the HuggingFace SAM3 PVS adapters, which present the SAM2 predictor interface
    so run_chain / per-slice / propagation / save_masks all run unchanged.
    """
    if cfg.backend == "sam3":
        from sam2_utils import sam3_backend
        ckpt = cfg.sam3_checkpoint or sam3_backend.DEFAULT_CHECKPOINT_DIR
        return sam3_backend.Sam3ImagePredictor(ckpt), sam3_backend.Sam3VideoPredictor(ckpt)
    if cfg.backend != "sam2":
        raise ValueError(f"unknown backend {cfg.backend!r}; expected 'sam2' or 'sam3'")
    image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image",
                                               image_size=cfg.image_size)
    video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video",
                                               image_size=cfg.image_size)
    return image_predictor, video_predictor
```

Then in BOTH `_build_session` and `_build_gt_session`, replace the two `setup.build_predictor` pairs with:

```python
    image_predictor, video_predictor = build_predictors(cfg)
```

- [ ] **Step 5: Run to verify pass**

Run: `py -3 -m pytest tests/test_backend_selection.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Add the CLI flags**

In `batch.py`'s argument parser (where `--preset` / `--model-size` are defined), add:

```python
    ap.add_argument("--backend", choices=["sam2", "sam3"], default=None,
                    help="segmentation backend (default: the preset's, which is sam2)")
    ap.add_argument("--sam3-checkpoint", default=None,
                    help="SAM3 HuggingFace checkpoint dir (default: sam3_backend.DEFAULT_CHECKPOINT_DIR)")
```

Then, where other CLI values override the preset-built `PipelineConfig` (the same block that applies `--model-size`, `--output-root`, etc.), thread these through, applying only when provided:

```python
    if args.backend is not None:
        cfg.backend = args.backend
    if args.sam3_checkpoint is not None:
        cfg.sam3_checkpoint = args.sam3_checkpoint
```

(Match the exact override mechanism already used in `main`; if config is frozen/rebuilt from a dict, set these keys the same way `--model-size` is set.)

- [ ] **Step 7: Verify default parity and commit**

Run: `py -3 -m pytest -q` (Expected: full suite passes, prior count + 3)
Run: `py -3 batch.py --help` (Expected: shows `--backend {sam2,sam3}` and `--sam3-checkpoint`; exits 0)
Run: `ruff check pipeline/config.py batch.py tests/test_backend_selection.py` (Expected: clean)

```bash
git add pipeline/config.py batch.py tests/test_backend_selection.py
git commit -m "feat(sam3): --backend {sam2,sam3} switch in batch.py (sam2 default unchanged)"
```

---

### Task 2: Local `--backend sam3` single-chain parity gate (GPU, manual)

Confirm SAM3 masks save to disk and score with `merge_metric` identically to SAM2, BEFORE any cluster work. The bake-off scored in-RAM; this exercises the real `save_masks` + on-disk scoring path.

**Files:**
- No code changes expected. If a real mask-format mismatch surfaces, fix it in `sam2_utils/sam3_backend.py` and note it.

- [ ] **Step 1: Run one target-worm chain through batch.py on SAM3**

Pick a short chain and a per-slice preset. Run to a scratch output root so nothing is overwritten:

```bash
py -3 batch.py --preset original --backend sam3 --neurons AIAL \
   --output-root E:/tmp/sam3_parity  # or any scratch path
```

Expected: completes without exception; writes per-chain masks + `_manifest.csv` under the scratch root.

- [ ] **Step 2: Score the SAM3 tree and confirm parity**

Run the merge-metric scorer over the scratch tree (the same `eval.merge_metric` entry the Phase-1 A/B used; confirm the exact CLI from `docs/reference/cli.md`). Expected: it reads the SAM3 masks and produces `foreign_node_rate` / `dropout` / `underfill` / `mild_bleed` without shape or dtype errors, i.e. the on-disk SAM3 masks are scored exactly like SAM2's.

- [ ] **Step 3: Record the parity result**

Note in `.git/sdd/` (or the report file) that on-disk save + score parity holds (or the fix that made it hold). No commit unless a code fix was needed; if so:

```bash
git add sam2_utils/sam3_backend.py
git commit -m "fix(sam3): <parity fix> so saved SAM3 masks score like SAM2"
```

---

### Task 3: Cluster wiring + pin the baseline presets

**Files:**
- Modify: `cluster/run_array.sh`
- Read/confirm: `sam2_utils/presets.py` (pin the two baseline preset names)

- [ ] **Step 1: Pin the baseline preset names**

Read `sam2_utils/presets.py` and identify the two presets that produced the SAM2 baseline trees: the per-slice winner (`original_perslice_only_guard` per the Phase-1 close-out) and the propagation baseline (the tier2 crop-1 + first-slice-forced + negatives config). Record the exact keys. If the propagation baseline has no named preset, add a thin one mirroring the config that produced the baseline tree, and commit it separately:

```bash
git add sam2_utils/presets.py
git commit -m "feat(presets): name the SAM2 propagation baseline preset for the SAM3 comparison"
```

- [ ] **Step 2: Forward backend/checkpoint/output-root through the array script**

In `cluster/run_array.sh`, add pass-through of the backend, checkpoint, and output-root to the `batch.py` invocation via environment variables the SBATCH script reads (matching how it already forwards the preset). For example, read `${SAM_BACKEND:-sam2}`, `${SAM3_CKPT:-}`, `${OUT_ROOT:-}` and append `--backend "$SAM_BACKEND"`, and `--sam3-checkpoint`/`--output-root` when set.

- [ ] **Step 3: Dry-run check**

Run a shell dry-run (e.g. `bash -n cluster/run_array.sh` for syntax, and echo the constructed `batch.py` command line for a sample chunk with `SAM_BACKEND=sam3`) to confirm the flags land correctly. Expected: the printed command contains `--backend sam3 --sam3-checkpoint <path> --output-root <path> --preset <name> --neurons <chunk>`.

- [ ] **Step 4: Commit**

```bash
git add cluster/run_array.sh
git commit -m "feat(cluster): forward --backend/--sam3-checkpoint/--output-root through the array job"
```

---

### Task 4: Narval runbook + doc updates

**Files:**
- Create: `docs/how-to/run-sam3-on-narval.md`
- Modify: `docs/CHANGELOG.md`, `docs/reference/code-map.md`, `docs/explanation/roadmap.md`

- [ ] **Step 1: Write the runbook**

Create `docs/how-to/run-sam3-on-narval.md` covering, as numbered steps a reader can follow from a fresh Narval shell:
1. Upload the checkpoint (`F:\sam3\huggingface`, ~3.4 GB) to a persistent project path (`scp`/`rsync`, or Globus).
2. Environment, lead with the fresh venv: `module load python`, `python -m venv`, `pip install --no-index torch` (cluster-matched), then `pip install transformers>=5.13`. Record the exact working module + versions. Document the Apptainer container as the fallback if the venv route fails on torch/CUDA (build the image from the known-good local stack, `apptainer exec`).
3. Smoke ONE chunk first (`--array=0-0`) to measure per-chain walltime and confirm the env loads and SAM3 runs; only then size and submit the full array.
4. Submit each of the two SAM3 configs (per-slice, propagation) with `SAM_BACKEND=sam3`, the checkpoint path, a distinct `OUT_ROOT`, and the matching preset.
5. Merge shards, download the two SAM3 trees, score locally with `eval.merge_metric` (or `eval/retro_eval.py`), and table the numbers next to the SAM2 baselines.
Include the honest caveats: SAM3 is 3 to 4x slower (allocation), `pred_iou` is NaN for SAM3 propagation, and SAM3 masks must not overwrite SAM2 trees.

- [ ] **Step 2: Update CHANGELOG, code-map, roadmap**

Add a CHANGELOG entry for the `--backend sam3` switch and the Phase-2 plan. Add the `--backend`/`--sam3-checkpoint` flags to the code-map's batch.py/entry-points notes. Add a roadmap line that the SAM3 whole-set cluster comparison is queued (following the bake-off's next-step note). Run the `humanizer` skill on all of this prose.

- [ ] **Step 3: Verify and commit**

Run: `grep -rnE "—|–" docs/how-to/run-sam3-on-narval.md docs/CHANGELOG.md docs/reference/code-map.md docs/explanation/roadmap.md` (Expected: no matches)
Run: `py -3 -m pytest -q` (Expected: passes)

```bash
git add docs/how-to/run-sam3-on-narval.md docs/CHANGELOG.md docs/reference/code-map.md docs/explanation/roadmap.md
git commit -m "docs(sam3): Narval runbook for the whole-set SAM3 cluster comparison + register backend switch"
```

---

## Self-review notes

- **Spec coverage:** backend switch (Task 1), config matching + output isolation (Task 1 `--output-root`, Task 3 preset pinning), cluster wiring (Task 3), Narval runbook with venv + container fallback (Task 4), mask-format parity gate (Task 2), SAM2-default byte-identical (Task 1 default + parity test). All covered.
- **Default safety:** `backend` defaults to `sam2`; the routing test asserts the sam2 path is unchanged, and Task 1 Step 7 confirms `--help` and the full suite.
- **Honesty:** the exact propagation-baseline preset name is pinned in Task 3 Step 1 against the real `presets.py` rather than guessed here; the Narval env specifics are discovered and recorded in the runbook rather than asserted.
