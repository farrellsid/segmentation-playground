# SAM3 PVS tracker bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure SAM2 versus SAM3 on the target worm across both segmentation strategies (propagation and per-slice) with an apples-to-apples 2x2 bake-off.

**Architecture:** Two thin SAM3 adapters present the exact predictor interfaces the existing `pipeline.segment_per_slice` and `pipeline.propagate` already call, so the SAM3 cells run through the same code as SAM2 with only the model swapped. A driver harness runs all four cells on shared chains and scores them with the existing GT-free merge-metric and membrane detectors. SAM2 stays the pipeline default; nothing in `batch.py` changes.

**Tech Stack:** Python 3.13, PyTorch 2.12 + CUDA 13, HuggingFace `transformers` 5.13.1 (`Sam3TrackerModel`, `Sam3TrackerProcessor`, `Sam3TrackerVideoModel`, `Sam3TrackerVideoProcessor`), numpy, existing `sam2_utils` / `pipeline` / `eval` modules.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages. Use commas, colons, parentheses, or separate sentences.
- Run the `humanizer` skill on any committed prose (docs, this plan's doc edits, commit bodies).
- No `Co-Authored-By: Claude` trailer in commits.
- Tests are CPU-only and torch-free for pure logic: `py -3 -m pytest`. Adapter code that needs a GPU and the checkpoint is smoke-tested manually by the harness, never in CI.
- Lint with `ruff check .`; clean only files you touch, do not reformat the tree.
- The library (`pipeline.py`, `sam2_utils/`) must never import drivers (`batch`, `gui`, `run_aval`) or `eval`. `tests/test_import_direction.py` enforces this. Therefore `sam2_utils/sam3_backend.py` MUST import `torch` and `transformers` lazily (inside functions/methods), so importing the module stays CPU-only and import-direction-clean.
- Commit incrementally, one concern per commit.
- SAM2 remains the default everywhere; this plan adds no `--backend` flag and does not touch `batch.py`.
- SAM3 loads from the local path `F:\sam3\huggingface` via `from_pretrained`, in bfloat16 with CPU offload, on the RTX 3050 6GB.
- All four cells run at the canonical `scale = 8` grid (`pipeline.config` default), so numbers compare to prior Phase 0/1/2 results.

---

## File structure

- Create `sam2_utils/sam3_backend.py`: the two adapters (`Sam3ImagePredictor`, `Sam3VideoPredictor`) plus torch-free helper functions for prompt formatting and output mapping. Imports torch/transformers lazily.
- Create `tests/test_sam3_backend.py`: CPU-only, torch-free tests for the helper functions.
- Create `experiments/sam3_bakeoff.py`: the driver harness (may import `eval` and `sam2_utils`; it is a driver, not library code).
- Create `docs/explanation/sam3-bakeoff-findings.md`: the recorded SAM3 API characterization (Task 1) and, later, the results table.
- Modify `docs/CHANGELOG.md` and `docs/reference/code-map.md`: record the new module and driver.

---

### Task 1: Characterize the SAM3 tracker API (spike, GPU)

Run tiny real SAM3 calls to lock down the exact working call sequences and output shapes, and answer the reverse-propagation gating question, BEFORE writing adapters. This task produces a findings document that later tasks copy call sequences from, so no later task guesses an API.

**Files:**
- Create: `experiments/sam3_probe.py` (throwaway spike script)
- Create: `docs/explanation/sam3-bakeoff-findings.md` (recorded findings)

**Interfaces:**
- Produces: a findings doc with verified, copy-pasteable call sequences for (a) image tracker predict, (b) video session init + add prompt + propagate, (c) whether reverse propagation from a mid-stack frame is supported, (d) exact output field names and tensor shapes, (e) peak VRAM for a short video.

- [ ] **Step 1: Write the image-tracker probe**

Create `experiments/sam3_probe.py`. Load the image tracker from the local checkpoint and run a single point prompt on a small synthetic RGB image (e.g. a 256x256 numpy array). Print the exact processor kwargs used, `type(outputs)`, the output field names (`dir(outputs)`), and the shapes of `pred_masks` and `iou_scores`. Print what `processor.post_process_masks(...)` returns and its shape.

```python
import numpy as np, torch
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
CKPT = r"F:\sam3\huggingface"
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3TrackerModel.from_pretrained(CKPT).to(dev)
proc = Sam3TrackerProcessor.from_pretrained(CKPT)
img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
inputs = proc(images=img, input_points=[[[[128, 128]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
with torch.no_grad():
    out = model(**inputs, multimask_output=True)
print("fields:", [k for k in out.keys()])
print("pred_masks:", tuple(out.pred_masks.shape), "iou_scores:", tuple(out.iou_scores.shape))
masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
print("post_process_masks type/shape:", type(masks), np.asarray(masks[0]).shape)
```

- [ ] **Step 2: Run the image probe and record results**

Run: `py -3 experiments/sam3_probe.py`
Record in `docs/explanation/sam3-bakeoff-findings.md`: the exact working image call sequence, output field names, and both shapes. If any kwarg name differs from the above, record the corrected one.

- [ ] **Step 3: Add the video-tracker probe (including reverse)**

Extend `experiments/sam3_probe.py` to build an 8-frame synthetic video (list of small RGB arrays), init a video session, add a box prompt on frame 4 (a mid-stack frame), then attempt to propagate BOTH forward and backward. Probe the real API names verified to exist (`init_video_session`, `add_inputs_to_inference_session`, `propagate_in_video_iterator`, `reset_inference_session`, `post_process_masks`). Print, for each yielded output, `frame_idx` and `pred_masks` shape. Explicitly attempt reverse propagation and capture whether it is supported (a kwarg, a separate call, or unavailable).

```python
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
vmodel = Sam3TrackerVideoModel.from_pretrained(CKPT).to(dev, dtype=torch.bfloat16)
vproc = Sam3TrackerVideoProcessor.from_pretrained(CKPT)
frames = [(np.random.rand(128, 128, 3) * 255).astype("uint8") for _ in range(8)]
sess = vproc.init_video_session(video=frames, inference_device=dev, dtype=torch.bfloat16)
# Record the EXACT signature that works for adding a box on frame 4 for obj_id 1,
# and how frames_idx / obj_ids / input_boxes must be shaped. Then:
for out in vmodel.propagate_in_video_iterator(sess):
    print("fwd frame", out.frame_idx, tuple(out.pred_masks.shape))
# Then probe reverse: try a reverse kwarg / start frame, and record the outcome verbatim.
```

- [ ] **Step 4: Run the video probe and record results + reverse verdict**

Run: `py -3 experiments/sam3_probe.py`
Record in the findings doc: the exact video call sequence, coordinate convention (pixel vs normalized, and whether `original_size` is required), output field names and shapes, the peak VRAM (`torch.cuda.max_memory_allocated()`), and a clear REVERSE PROPAGATION verdict: supported (how) or not (so the adapter needs the dual-session or frame-reindex workaround). This verdict gates Task 5.

- [ ] **Step 5: Commit the findings**

```bash
git add docs/explanation/sam3-bakeoff-findings.md experiments/sam3_probe.py
git commit -m "spike(sam3): characterize HF tracker image+video API, record reverse-propagation verdict"
```

---

### Task 2: Torch-free prompt-formatting helpers

Pure functions that convert the pipeline's prompt arrays into the nested-list shapes the SAM3 processors expect. No torch, no model. Fully TDD.

**Files:**
- Create: `sam2_utils/sam3_backend.py`
- Test: `tests/test_sam3_backend.py`

**Interfaces:**
- Produces:
  - `to_image_prompts(point_coords: np.ndarray | None, point_labels: np.ndarray | None, box: np.ndarray | None) -> dict` returning a dict with keys among `input_points`, `input_labels`, `input_boxes` shaped for one image and one object, or an empty value where a prompt is absent. Shapes follow Task 1's recorded convention: points nested as `[[[ [x,y], ... ]]]`, labels as `[[[l, ...]]]`, a single box as `[[[x1,y1,x2,y2]]]`.
  - `to_video_prompt(frame_idx: int, obj_id: int, box: np.ndarray | None, points: np.ndarray | None, labels: np.ndarray | None) -> dict` returning the kwargs dict for the session add-prompt call, shaped per Task 1.

- [ ] **Step 1: Write failing tests for `to_image_prompts`**

```python
import numpy as np
from sam2_utils import sam3_backend as sb

def test_to_image_prompts_points_only():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    labs = np.array([1, 0])
    out = sb.to_image_prompts(pts, labs, None)
    assert out["input_points"] == [[[[10.0, 20.0], [30.0, 40.0]]]]
    assert out["input_labels"] == [[[1, 0]]]
    assert "input_boxes" not in out

def test_to_image_prompts_box_only():
    box = np.array([1.0, 2.0, 3.0, 4.0])
    out = sb.to_image_prompts(None, None, box)
    assert out["input_boxes"] == [[[1.0, 2.0, 3.0, 4.0]]]
    assert "input_points" not in out

def test_to_image_prompts_points_and_box():
    out = sb.to_image_prompts(np.array([[5.0, 6.0]]), np.array([1]), np.array([0.0, 0.0, 9.0, 9.0]))
    assert out["input_points"] == [[[[5.0, 6.0]]]]
    assert out["input_labels"] == [[[1]]]
    assert out["input_boxes"] == [[[0.0, 0.0, 9.0, 9.0]]]
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3 -m pytest tests/test_sam3_backend.py -v`
Expected: FAIL (module or function not defined).

- [ ] **Step 3: Implement `to_image_prompts`**

In `sam2_utils/sam3_backend.py` (module docstring must state the lazy-import rule):

```python
"""SAM3 (HuggingFace transformers) adapters presenting the SAM2 predictor interface.

torch and transformers are imported LAZILY inside methods so importing this module
stays CPU-only and does not violate the library import-direction rule.
"""
from __future__ import annotations

import numpy as np


def to_image_prompts(point_coords, point_labels, box):
    """Shape pipeline prompt arrays into SAM3 image-processor kwargs (one image, one object)."""
    out: dict = {}
    if point_coords is not None and len(point_coords):
        pts = np.asarray(point_coords, dtype=float).tolist()
        labs = np.asarray(point_labels, dtype=int).tolist()
        out["input_points"] = [[pts]]
        out["input_labels"] = [[labs]]
    if box is not None:
        b = np.asarray(box, dtype=float).ravel().tolist()
        out["input_boxes"] = [[b]]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `py -3 -m pytest tests/test_sam3_backend.py -v`
Expected: PASS (the three `to_image_prompts` tests).

- [ ] **Step 5: Write failing tests for `to_video_prompt`, then implement it**

Add tests asserting the shape recorded in Task 1 (adjust literals to match the findings doc if Task 1 corrected them):

```python
def test_to_video_prompt_box():
    out = sb.to_video_prompt(4, 1, np.array([1.0, 2.0, 3.0, 4.0]), None, None)
    assert out["frame_idx"] == 4
    assert out["obj_ids"] == 1
    assert out["input_boxes"] == [[[1.0, 2.0, 3.0, 4.0]]]

def test_to_video_prompt_points():
    out = sb.to_video_prompt(0, 2, None, np.array([[7.0, 8.0]]), np.array([1]))
    assert out["frame_idx"] == 0
    assert out["obj_ids"] == 2
    assert out["input_points"] == [[[[7.0, 8.0]]]]
    assert out["input_labels"] == [[[1]]]
```

Implement `to_video_prompt` to return exactly those kwargs (names taken from Task 1's recorded `add_inputs_to_inference_session` signature).

- [ ] **Step 6: Run tests, then commit**

Run: `py -3 -m pytest tests/test_sam3_backend.py -v` (Expected: PASS)
Run: `ruff check sam2_utils/sam3_backend.py tests/test_sam3_backend.py` (Expected: clean)

```bash
git add sam2_utils/sam3_backend.py tests/test_sam3_backend.py
git commit -m "feat(sam3): torch-free prompt-formatting helpers for SAM3 adapters"
```

---

### Task 3: Torch-free output-mapping helpers

Pure functions that turn already-detached numpy outputs into the shapes the pipeline expects, so the model-touching adapters stay thin. No torch.

**Files:**
- Modify: `sam2_utils/sam3_backend.py`
- Test: `tests/test_sam3_backend.py`

**Interfaces:**
- Produces:
  - `select_image_masks(masks: np.ndarray, scores: np.ndarray, low_res_logits: np.ndarray, multimask_output: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(masks, scores, logits)` in the same layout `SAM2ImagePredictor.predict` returns: `masks` shape `(num_masks, H, W)` bool, `scores` shape `(num_masks,)`, `logits` shape `(num_masks, h, w)`. When `multimask_output` is False, num_masks is 1.
  - `video_logits_to_mask(logit_hw: np.ndarray) -> np.ndarray` returning `logit_hw > 0.0` as bool, matching `PropagationSession._collect`'s threshold.

- [ ] **Step 1: Write failing tests**

```python
def test_select_image_masks_multimask_keeps_all():
    masks = np.zeros((3, 4, 4), bool); scores = np.array([0.1, 0.9, 0.5])
    logits = np.zeros((3, 2, 2), float)
    m, s, lg = sb.select_image_masks(masks, scores, logits, multimask_output=True)
    assert m.shape == (3, 4, 4) and s.shape == (3,) and lg.shape == (3, 2, 2)

def test_select_image_masks_single_returns_one():
    masks = np.zeros((1, 4, 4), bool); scores = np.array([0.7]); logits = np.zeros((1, 2, 2), float)
    m, s, lg = sb.select_image_masks(masks, scores, logits, multimask_output=False)
    assert m.shape == (1, 4, 4) and s.shape == (1,) and lg.shape == (1, 2, 2)
    assert m.dtype == bool

def test_video_logits_to_mask_thresholds_at_zero():
    lg = np.array([[-1.0, 0.5], [2.0, -0.1]])
    assert sb.video_logits_to_mask(lg).tolist() == [[False, True], [True, False]]
```

- [ ] **Step 2: Run to verify failure**

Run: `py -3 -m pytest tests/test_sam3_backend.py -v`
Expected: FAIL for the three new tests.

- [ ] **Step 3: Implement the helpers**

```python
def select_image_masks(masks, scores, low_res_logits, multimask_output):
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float).ravel()
    logits = np.asarray(low_res_logits, dtype=float)
    if masks.ndim == 2:
        masks = masks[None]
    if logits.ndim == 2:
        logits = logits[None]
    return masks, scores, logits


def video_logits_to_mask(logit_hw):
    return np.asarray(logit_hw, dtype=float) > 0.0
```

- [ ] **Step 4: Run tests, lint, commit**

Run: `py -3 -m pytest tests/test_sam3_backend.py -v` (Expected: PASS all)
Run: `ruff check sam2_utils/sam3_backend.py tests/test_sam3_backend.py` (Expected: clean)

```bash
git add sam2_utils/sam3_backend.py tests/test_sam3_backend.py
git commit -m "feat(sam3): torch-free output-mapping helpers for SAM3 adapters"
```

---

### Task 4: `Sam3ImagePredictor` adapter (GPU smoke)

Wrap `Sam3TrackerModel` behind the `SAM2ImagePredictor` surface that `pipeline.predict.image_predict` calls: `set_image()` and `predict(point_coords, point_labels, box, multimask_output) -> (masks, scores, logits)`. Uses Task 2/3 helpers and Task 1's recorded call sequence.

**Files:**
- Modify: `sam2_utils/sam3_backend.py`
- Verify with: `experiments/sam3_probe.py` extended to drive the adapter (manual, GPU)

**Interfaces:**
- Consumes: `to_image_prompts` (Task 2), `select_image_masks` (Task 3), Task 1 image call sequence.
- Produces: class `Sam3ImagePredictor(checkpoint_dir: str, device=None)` with:
  - `set_image(self, image_rgb: np.ndarray) -> None`
  - `predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]` matching `SAM2ImagePredictor.predict`'s `(masks (N,H,W) bool, scores (N,), logits (N,h,w))`.

- [ ] **Step 1: Implement the adapter class**

Import torch/transformers lazily inside `__init__`/methods. Store the RGB image and its `(H, W)` on `set_image`. In `predict`, build processor inputs from `to_image_prompts`, run the model under `torch.no_grad()` using the exact sequence recorded in Task 1, call `post_process_masks` with `original_sizes=[(H, W)]` for full-res masks, detach `iou_scores` and the low-res `pred_masks` to numpy, and return `select_image_masks(...)`. Do not normalize coordinates (Task 1 confirmed the image tracker uses pixel coordinates).

```python
class Sam3ImagePredictor:
    def __init__(self, checkpoint_dir: str, device=None):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3TrackerModel.from_pretrained(checkpoint_dir).to(self.device)
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint_dir)
        self._image = None
        self._hw = None

    def set_image(self, image_rgb):
        self._image = np.asarray(image_rgb)
        self._hw = (int(self._image.shape[0]), int(self._image.shape[1]))

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        torch = self._torch
        prompts = to_image_prompts(point_coords, point_labels, box)
        inputs = self.processor(images=self._image, return_tensors="pt", **prompts).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=multimask_output)
        H, W = self._hw
        full = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = np.asarray(full)                                  # (N, H, W) or (1, N, H, W): squeeze in helper
        scores = out.iou_scores.detach().cpu().numpy()
        low = out.pred_masks.detach().float().cpu().numpy()
        masks = masks.reshape(-1, H, W)
        return select_image_masks(masks, scores.reshape(-1), low.reshape(low.shape[-3], low.shape[-2], low.shape[-1]), multimask_output)
```

Note: if Task 1 recorded different field names or a different `post_process_masks` return nesting, adjust these lines to match the findings doc before running.

- [ ] **Step 2: Smoke-test the adapter through `image_predict`**

Extend `experiments/sam3_probe.py` to build a `Sam3ImagePredictor`, wrap a real target-worm frame (load one crop image via `cv2`), build a `Prompts` with one positive point, and call `pipeline.predict.image_predict(adapter, image, prompts, multimask=True, select_area_bounds=(1e-5, 0.4))`. Assert it returns a bool mask of the frame shape and a finite score.

Run: `py -3 experiments/sam3_probe.py`
Expected: prints a mask shape equal to the frame and a float score; no exception.

- [ ] **Step 3: Commit**

```bash
git add sam2_utils/sam3_backend.py experiments/sam3_probe.py
git commit -m "feat(sam3): Sam3ImagePredictor adapter over Sam3TrackerModel (SAM2 image API)"
```

---

### Task 5: `Sam3VideoPredictor` adapter (GPU smoke)

Wrap `Sam3TrackerVideoModel` behind the surface `PropagationSession` calls. Reverse handling follows Task 1's verdict.

**Files:**
- Modify: `sam2_utils/sam3_backend.py`
- Verify with: `experiments/sam3_probe.py` (manual, GPU)

**Interfaces:**
- Consumes: `to_video_prompt` (Task 2), `video_logits_to_mask` (Task 3), Task 1 video sequence + reverse verdict.
- Produces: class `Sam3VideoPredictor(checkpoint_dir, device=None)` with methods matching what `PropagationSession` uses:
  - `init_state(self, video_path: str, offload_video_to_cpu: bool = True, **kw) -> object` (loads `{idx:05d}.jpg` frames from the dir into a session; returns the session object as the "inference_state")
  - `reset_state(self, inference_state) -> None`
  - `add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None, clear_old_points=False)`
  - `add_new_mask(self, inference_state, frame_idx, obj_id, mask)`
  - `propagate_in_video(self, inference_state, reverse=False, start_frame_idx=None, max_frame_num_to_track=None) -> Iterator[tuple[int, list[int], list]]` yielding `(frame_idx, [obj_id], [logit_hw_array])` so `PropagationSession._collect` can index `[i].cpu().numpy()`; the yielded logit object exposes `.cpu().numpy()` (return a torch tensor at frame resolution, obtained via `post_process_masks(..., binarize=False)`).

- [ ] **Step 1: Implement construction, frame loading, seeding, reset**

Lazy-import torch/transformers. `init_state` reads the sorted `{idx:05d}.jpg` frames from `video_path` with `cv2` into a frames list and calls `init_video_session(video=frames, inference_device=self.device, dtype=torch.bfloat16, ...)` (store any storage/processing device kwargs Task 1 recorded for CPU offload). `add_new_points_or_box` / `add_new_mask` call `add_inputs_to_inference_session` with `to_video_prompt(...)` kwargs. `reset_state` calls `inference_state.reset_inference_session()`.

- [ ] **Step 2: Implement `propagate_in_video` with the reverse handling from Task 1**

Iterate `self.vmodel.propagate_in_video_iterator(inference_state, ...)`, and for each output run `post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]` to get frame-resolution logits, then yield `(int(out.frame_idx), [obj_id], [logit_tensor])`. Implement `reverse` per Task 1: if the iterator takes a `reverse`/`start_frame_idx` kwarg, forward it; if reverse is unsupported, implement the recorded workaround (for the bake-off, seed both directions or reindex frames) and document it in a code comment referencing the findings doc.

- [ ] **Step 3: Smoke-test through `PropagationSession` on the short chain**

Extend `experiments/sam3_probe.py` (or add a `--video` branch) to build a `Sam3VideoPredictor`, then run `pipeline.propagate.propagate(adapter, frames_dir, prompts, anchor_frame_idx, obj_id=1)` on the SHORT chain's prepared frames. Assert it returns `(video_segments, frame_conf, pred_iou)` with one mask per frame.

Run: `py -3 experiments/sam3_probe.py --video`
Expected: a `video_segments` dict covering the chain's frames, each a bool mask; no exception. Confirms reverse works (or the workaround does).

- [ ] **Step 4: Commit**

```bash
git add sam2_utils/sam3_backend.py experiments/sam3_probe.py
git commit -m "feat(sam3): Sam3VideoPredictor adapter over Sam3TrackerVideoModel (SAM2 video API)"
```

---

### Task 6: Bake-off harness (the 2x2 comparison)

Run all four cells on the two chains, score with the existing metrics, and emit overlays plus a table.

**Files:**
- Create: `experiments/sam3_bakeoff.py`
- Modify: `docs/explanation/sam3-bakeoff-findings.md` (append the results table)

**Interfaces:**
- Consumes: `sam2_utils.setup` (SAM2 build), `sam2_utils.sam3_backend` (SAM3 adapters), `pipeline.propagate.propagate`, `pipeline.propagate.segment_per_slice`, `pipeline.predict.build_prompts`, `review.load_chain`, `pipeline.load_state`, `eval.merge_metric`, `sam2_utils.membrane`, `pipeline.config`.

- [ ] **Step 1: Implement the harness skeleton and CLI**

Create `experiments/sam3_bakeoff.py` with a CLI: `--chains` (default `AIAL:5,AIAL:0`), `--root` (default `config.OUTPUT_ROOT`), `--checkpoint` (default `F:\sam3\huggingface`), `--scale` (default 8), `--out` (default `docs/figures/sam3-bakeoff`). For each chain, load it (the `coprop_lab.load_lab_chain` / `review.load_chain` pattern), build anchor prompts once via `predict.build_prompts`, and define the four cells:
- `sam2_prop`: build SAM2 video predictor (`setup.build_predictor(kind="video")`), run `propagate(...)`.
- `sam2_perslice`: build SAM2 image predictor, run `segment_per_slice(...)`.
- `sam3_prop`: build `Sam3VideoPredictor`, run the SAME `propagate(...)`.
- `sam3_perslice`: build `Sam3ImagePredictor`, run the SAME `segment_per_slice(...)`.

Fail-fast rule: run the short chain's `sam3_prop` cell first; if it raises on reverse propagation, print a clear message and stop before the long chain.

- [ ] **Step 2: Score each cell and write overlays**

For each cell's `video_segments`, compute the merge-metric (foreign-node containment + dropout via `eval.merge_metric`) and the membrane detectors (underfill, mild-bleed via `sam2_utils.membrane`), using the SAME scoring call for all four. Save per-cell overlay PNGs to `--out`. Track wall-clock and `torch.cuda.max_memory_allocated()` per cell; on a CUDA OOM, record the cell as "Narval-only" and continue.

- [ ] **Step 3: Emit the 4-row table and run it**

Print and append to `docs/explanation/sam3-bakeoff-findings.md` a table with rows `{sam2,sam3} x {prop,perslice}` and columns: foreign_node_rate, dropout, underfill, mild_bleed, seconds, peak_vram_gb.

Run: `py -3 experiments/sam3_bakeoff.py --chains AIAL:5`
Expected: the short chain completes (or fail-fast message on reverse), a 4-row table prints.
Then run the full default: `py -3 experiments/sam3_bakeoff.py`

- [ ] **Step 4: Commit the harness (code only; figures per repo convention)**

```bash
git add experiments/sam3_bakeoff.py
git commit -m "feat(sam3): 2x2 bake-off harness (SAM2 vs SAM3, propagation + per-slice)"
```

---

### Task 7: Record results and update docs

**Files:**
- Modify: `docs/explanation/sam3-bakeoff-findings.md` (results + interpretation)
- Modify: `docs/CHANGELOG.md`
- Modify: `docs/reference/code-map.md`

- [ ] **Step 1: Write the findings interpretation**

In `docs/explanation/sam3-bakeoff-findings.md`, state whether SAM3 beat SAM2 on propagation, on per-slice, or both, with the merge-metric and membrane numbers, and note the reverse-propagation outcome and any memory limits found. Run the `humanizer` skill on this prose.

- [ ] **Step 2: Update CHANGELOG and code-map**

Add a CHANGELOG entry dated 2026-07-21 summarizing the SAM3 bake-off and its verdict. Add `sam2_utils/sam3_backend.py` and `experiments/sam3_bakeoff.py` to `docs/reference/code-map.md`. Run the `humanizer` skill on both edits. No em dashes.

- [ ] **Step 3: Verify suite and lint, then commit**

Run: `py -3 -m pytest` (Expected: PASS, including `tests/test_import_direction.py`, confirming `sam2_utils.sam3_backend` imports without torch)
Run: `ruff check .` (Expected: clean on touched files)

```bash
git add docs/explanation/sam3-bakeoff-findings.md docs/CHANGELOG.md docs/reference/code-map.md
git commit -m "docs(sam3): record bake-off results and register the new module + driver"
```

---

## Self-review notes

- **Spec coverage:** 2x2 matrix (Tasks 4, 5, 6), adapters through existing pipeline functions (Tasks 4, 5), merge-metric + membrane scoring only (Task 6), short + long chains with fail-fast (Task 6), HF transformers local load + bf16 + offload (Tasks 1, 4, 5), SAM2 default untouched (no `batch.py` edit anywhere), reverse-propagation risk (Task 1 verdict gates Task 5), pred_iou-NaN acceptable (not required by scoring in Task 6), memory reporting (Task 6). All covered.
- **Import direction:** `sam2_utils/sam3_backend.py` imports torch/transformers lazily; Task 7 Step 3 explicitly runs `test_import_direction.py` to confirm.
- **Placeholder honesty:** the model-touching kwargs are gated on Task 1's recorded findings rather than guessed; pure helpers carry complete test + implementation code.
