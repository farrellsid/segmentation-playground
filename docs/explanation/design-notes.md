# Design notes and backlog (detailed)

This is the original long-form design reference. The lean, current-state version now lives in
[architecture.md](architecture.md), and the load-bearing decisions are written up as one-page
records in [the ADRs](../adr/README.md). This file is kept for two things a newcomer may still want:
the detailed rationale behind specific knobs and defaults, and the active backlog of open work (the
numbered items in the backlog section below). For the build history, see
[../CHANGELOG.md](../CHANGELOG.md).

The text below predates the documentation reorganization, so it still uses the project's old
milestone and section vocabulary.

---

## 1. Goal

Segment ~300 *C. elegans* neurons (a few thousand maximal-linear-chains, MLCs) from an EM
`.tif` stack (~300 z-slices) into per-neuron mask volumes, for export to **Blender**.
Everything runs **locally** on one Windows + GPU box. Keep infrastructure simple: filesystem
only, no database, no server, no web app.

The pipeline is **semi-automatic**: the machine does the work, the human only reviews and
corrects what the machine flags.

---

## 2. Current state (June 2026)

**We are at milestone 4.** Milestones 1-3.5 are done; the M4 napari GUI core has landed and had
its first real-use pass. The next milestone is **M4.5** (the predictor model + label-gated
accuracy work that consumes the labels M4 collects).

What exists and works today:

- **Library + thin drivers.** `pipeline.py` is a pure library (phase functions, `PipelineConfig`,
  `ChainState`, `run_chain`, `PropagationSession`, `run_qc`). Drivers on top: `run_aval.py` (one
  chain / M1 regression harness, reproduces the notebook pixel-for-pixel), `batch.py` (headless
  batch + manifest/resume/triage), `gui.py` (napari review/correction). See §3.
- **Inline QC + flagging.** Every chain is auto-scored after save (`run_qc`) and marked
  `done`/`flagged`; only flagged frames reach a human. Four signals: `area_ratio`, `temporal_iou`,
  `skeleton_contained`, `pred_iou`. Thresholds are `qc_*` knobs on `PipelineConfig`.
- **Headless batch.** Validated on a 24-neuron / 384-chain subset; `_manifest.csv` + resume +
  `_triage.csv` + `_timing.csv`, 0 errors. The triage queue is gated on `intervene` (≥2 signals).
- **Anchor-quality hardening (M3.5).** High-res anchor crop is the default image phase; an
  observational anchor gate (`score_anchor`); tier-2 per-chain propagation crop landed (default-off
  globally, **auto-on as a second pass for flagged chains**, regression-free via an `image_score`
  fallback). Optional model-free `postprocess_mask` and seed-shape knobs (all default-off,
  defaults reproduce M1).
- **M4 GUI core.** `gui.py` + `review_queue.py` (work queue + GUI-owned `_review.csv` ledger) +
  `labels.py` (per-frame label engine = the M4.5 training data). Open a flagged chain, scrub to
  flagged frames, edit prompt points, paint an anchor mask, re-run the image phase, resume
  propagation (`PropagationSession`), and rewrite `masks/` + `qc.csv` + `state.json` so a corrected
  chain is byte-indistinguishable from a fresh batch run. GPU is lazy.

Known caveats carried into the backlog (§8): the displayed/stored mask is scale-8 (genuinely
higher-res masks need the tier-2 crop, M4.5); the GUI had real-use bugs (notably a reverse-resume
start bug); error detection is the acknowledged weakest part of the system; and a full-dataset
run (the remaining M3 confidence check) is still outstanding.

**Research step-back (June 2026).** Before committing M4.5 effort, the project took a deliberate step
back. A **cross-worm ground-truth dataset** (**SEM-Dauer 1**, a different worm from the target
"sensory ablated dauer", with matching EM and confirmed-segment markers) was obtained, unlocking real
**evaluation** and **finetuning** for the first time, since
almost everything in the M4.5 backlog had been "label-gated", and a broad SOTA survey was run. The
resulting evidence-backed roadmap is in **[`roadmap.md`](roadmap.md)**. Headline:
fix the evaluation metric (ERL + split/merge VOI) *before* further accuracy tuning; then finetune SAM2
on the new GT, and build a dense-segmentation + cross-z-linking hedge for the branch/merge failure.
**Stage 0 status (June 2026):** the ruler is built (`eval/`); a degenerate dry run (`predict_gt.py`,
small model) gave the first numbers and showed the skel→GT **coordinate transform is the keystone**
(~50% of slices miss with displaced masks; self-consistency ERL = 0 on perfect GT). Stage 0 now
finishes in four sub-steps: **0.1** ✅ registration upgraded to a **per-section affine** (residual
19.6→4.7 px, on-mask 67.9→85.7%; right-worm provenance confirmed), interactive GUI gut-check still
open; **0.2** score the **real `batch.py`** on SEM-Dauer 1 via an argparse dataset override (supersedes
`predict_gt.py`) ✅ **built + measured** (the real `batch.py` runs on SEM-Dauer 1 via a worm-agnostic
`FrameStore` seam + `eval.gt_dataset`; configurable subset; scored by `eval.score_batch` +
`eval.score_labelmap` for **region IoU + VOI/ARAND + per-neuron ERL**, with progress + timing +
`measurement_log.jsonl`. 3-neuron smoke: small→large+tier-2 micro-IoU 0.022→0.024, VOI 0.875→0.847,
ARAND ~0.16, ERL ~1%. Findings: large model alone barely helps (domain gap, not capacity); tier-2
helped where it engaged, VA4 kept `_pcrop`, IoU ~doubled, but 2/3 fell back at the 0.70 crop-score
floor (mis-calibrated cross-worm); merge/bleed-dominated. A floor, not the gate, next: multi-chain
neuron + lowered tier-2 floor + Stage-1 accuracy levers);
**0.3** ✅ full-res GT export done (`full_scale/`, 9728×9216, `GT_DOWNSCALE=1`) +
full-res `registration.json` done (¼ fit ×4 via `eval.scale_registration`, A≈I, on-mask 91.7%);
**0.4** ✅ ERL merge
tolerance (`--merge-tol-frac/-count`). See FUTURE_DIRECTIONS §5 Stage 0.

Note on why two registrations differ: SEM-Dauer 1 needs a *per-section* affine because its VAST masks/EM
are a **section-by-section realignment** (`SEM_dauer_1_vast_realignment_export`) of a stack the p280
skeleton was traced *before*, a cross-realignment mismatch. The sensory-ablated-dauer (project 336)
pipeline uses a **single global affine** (fit at one z) because its skeleton and raw EM share one
alignment with no such per-section realignment. The two situations are structurally different, so
SEM-Dauer 1's per-section need does **not** imply the target worm's single-z affine is wrong, **verified** (June 2026, `experiments/diag_target_registration.py`): across the 4 output neurons
(1603 frames, z 1367-1628), the cross-chain-coherent node↔mask-centroid offset stays ~flat with z
(corr −0.08) and is just 2 px at the far end (335 sections from the z=1293 fit point), i.e. the global
affine is z-stable, the few-px per-frame scatter is propagation/thinness, not registration drift.

---

## 3. Target architecture

**Not** "notebook → script." The move is **notebook → library + thin driver**.

### 3a. Library of phase functions
Each pipeline step is a small, testable function/class, `select_anchor`, `build_prompts`,
`image_predict`/`anchor_crop_predict`, `box_from_mask`, `prepare_video_frames`, `propagate`,
`postprocess_mask`, `save_masks`, `run_qc`, `aggregate`, pure-ish and reused by every driver.

### 3b. Per-chain state machine
An orchestrator carries a **serializable per-chain `ChainState`** and decides which phase runs next:

```
ChainState = {
    neuron, chain_idx, status,            # pending / running / done / flagged / failed
    anchor_frame_idx,
    prompts: {points, labels, box},
    image_mask_ref, qc_summary,
    triage_frames: [...],                 # frames needing human review
    crop_window,                          # set for tier-2 (_pcrop) chains
}
```
State serializes to disk so a chain can be paused, resumed after a crash, or re-opened in the GUI
without recomputation.

### 3c. Interruptible propagation, `PropagationSession` (landed)
A session object owns the live `inference_state` and exposes `seed()` /
`propagate(reverse=, start_frame_idx=, max_frames=)` (a lazy generator of `FrameResult` you can
`break`) / `add_points()` / `add_mask()` / `close()`. Continuity lives in `inference_state`: break
at a degrading frame, correct it, then `propagate(start_frame_idx=f)` resumes over the *same*
mutated state (never `reset_state`, that wipes prompts). It is torch/napari-free and is the
primitive both the headless loop and the M4 GUI drive. The headless `propagate()` is a thin
forward-then-reverse wrapper over it, so AVAL still reproduces pixel-for-pixel.

### 3d. Thin drivers (all call the same library)
Headless **batch** (primary), the **notebook** (exploration/debug), the **napari GUI** (review &
correction of the triage queue), and the **GT eval** (`eval.score_batch`, Stage 0). Run configs are
named **presets** in `sam2_utils/presets.py`, `batch.py --preset eval|original [--neurons …]` (and
`score_batch --preset eval`) bundle the worm/dataset, paths, model, tier-2/gif, and default neurons so a
run is one short command; any flag overrides. The `eval` preset retargets the *same* `run_chain` at
SEM-Dauer 1 via a `pipeline.FrameStore` seam (EM source) + registration-baked prompts, no per-chain
code changes; the default `original` preset is the target worm and stays byte-identical.

### 3e. Storage layout (filesystem, indexed)
```
output/
  _manifest.csv               # every chain × execution status, drives batch + resume
  _triage.csv                 # queued (intervene) frames across all chains, feeds the GUI
  _review.csv                 # GUI-owned review-status ledger (separate from _manifest)
  _timing.csv                 # per-chain phase seconds + peak VRAM
  <neuron>/chain_NN/
    state.json                # the ChainState above
    qc.csv                    # per-frame metrics (+ a `queue` column)
    masks/mask_<z:04d>.png    # canonical space (see §5)
  <neuron>/neuron_mask/mask_<z:04d>.png   # aggregated per-z union (M5, → Blender)

frames_root/                  # SAM2 JPEG frames, separate tree from output/
  frames_cache_s<scale>/z<file_z>.jpg     # shared decode cache; each frame written once
  chain_views/<neuron>_chain<idx>_s<scale>/00000.jpg ...   # 0-indexed links into cache
```

---

## 4. Cross-cutting principles

- **Automation-first, triage-second.** With thousands of chains, auto-run + QC is the only viable
  path; the human is a scarce resource spent only on flagged frames.
- **Supervision-tolerant; the win is propagation, not anchor perfection.** Even at one
  human-approved anchor per chain, the pipeline is ~50× faster than hand-painting slice by slice.
  Segmentation-accuracy gains (better prompts, finetuned models) are *optimizations*, judged by
  whether they shrink the triage queue enough to justify their complexity, not by chasing 100%
  auto-correctness. **Confirmed with the lab supervisor (June 2026):** full automation is *not*
  expected; accuracy with a human in the loop is the goal, and success is simply being faster than
  hand-colouring. This makes **accuracy rank above automation-%**, cropping and human-anchor
  fallbacks outrank raw speed.
- **One triage queue, one review tool.** Anchor-mask review (image phase) and mid-propagation
  degradation (video phase) are the *same* problem, a flagged frame that needs prompt edits. Don't
  build two GUIs.
- **Checkpoint everything per chain.** Resume on crash; never recompute a finished chain.
- **Centralize coordinate transforms.** Every space conversion lives in `alignment.py`; tag
  variables with their space suffix (`_tif`, `_sam`, `_crop`, `_pcrop`, `_cm`). See §5.
- **Measure, don't trust (the "ruler").** Invest in an accuracy lever only after measuring that it
  shrinks the queue. **And: don't ship decisions on assistant/AI "vibes", ground them in outside
  research and prior art** (mark recollection as to-verify). This discipline drives §9.

---

## 5. Invariants & gotchas (coordinate / filename / mask-space)

The durable facts to respect. (Full resolution stories for the issues that *created* these rules
are in [`CHANGELOG.md` old §5](../CHANGELOG.md#old-5).)

**Coordinate spaces** (all conversions go through `alignment.py`):
- `_tif` full-res stack px · `_sam` = `_tif / scale` (SCALE=8; the video-propagation input space
  **and** the canonical on-disk mask space) · `_crop` high-res anchor crop · `_pcrop` per-chain
  tier-2 crop · `_cm` CATMAID px · CATMAID nm · file-z vs CATMAID-z (`± FILE_Z_OFFSET`).
- `CropWindow` is the single home for `_crop`/`_pcrop` ↔ `_tif` ↔ `_sam`; the only row/col
  (`[y,x]` vs `(x,y)`) swap is isolated to `CropWindow.slice_tif`.

**Mask space / filenames:**
- Masks are **0/255 uint8 single-channel** PNGs named `mask_<catmaid_z:04d>.png`, stored at `_sam`
  with `save_downscale == scale == 8` (no resample). `run_qc` **hard-guards** `scale == save_downscale`
  (skipped only in crop mode, where the window remaps nodes instead).
- Do **not** confuse `pipeline.save_masks` (0/255 uint8) with `qc.save_masks` (uint16 *instance
  labels*, foreground == obj_id). The single-object pipeline uses the former; instance-label
  encoding is a multi-object (M5) concern.
- Tier-2 chains store masks in `_pcrop` and persist the `CropWindow` to `state.json`, so QC /
  `review` / the GUI rebuild the crop space; containment radius is rescaled by `scale/crop_scale`.

**Operational gotchas:**
- **`skeleton_contained` must use *this chain's* nodes**, not the whole neuron's (a multi-chain
  neuron's centroid sits off any single process → false 100%-flag). It is **tri-state** (True /
  False / NaN-when-no-node-at-that-z); only explicit `False` flags.
- **Manifest is append-mode.** A `qc_*`/`gate_*` threshold change mid-campaign silently mixes
  configs in `_manifest.csv`. **Clear or re-score** after any threshold change; sanity-check by
  confirming `min(area_frac among area-FAILs) > max(area_frac among PASS)`.

**Still open (deferred to M5):**
- **Anisotropy for Blender.** Voxels are 2/2/50 nm; at SCALE=8 that's ~16/16/50 nm (z ≈ 3× xy).
  The mesher must receive correct z spacing or the neuron is squashed in z.

---

## 6. Milestone roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 1 | **Refactor → library + state machine + serialization** | ✅ **Done.** Phase fns + `run_chain` + `ChainState`/`state.json`; reproduces AVAL pixel-for-pixel. Interruptible propagation landed as `PropagationSession` (§3c). |
| 2 | **Inline QC + flagging** | ✅ **Done.** `run_chain` step 9 = `run_qc` → `qc.csv` + status. Thresholds are `qc_*` knobs. `pred_iou` (4th signal) now populated. Read-only `review` viewer added. |
| 3 | **Headless batch runner + resume** | ✅ **Done (subset-validated).** `batch.py`: manifest, resume, `_triage.csv`, `clean`/`neurons` knobs, runtime telemetry. **Remaining:** the full-dataset run (see §8 Theme D). |
| 3.5 | **Headless anchor-quality hardening (pre-GUI)** | ✅ **Done as a pre-GUI phase.** Default anchor crop, observational anchor gate, tier-2 crop (auto second-pass on flagged), `multimask_anchor`/`seed_negatives`/`postprocess_mask` (default-off). Label-gated levers (tier-2 for real `noskel`, `pred_iou` floor, `gate_max_area_frac`) moved to M4.5. |
| 4 | **napari review/triage GUI** | ✅ **Core landed; first real-use pass done.** `gui.py` + `review_queue.py` + `labels.py`. One tool: scrub flagged frames, edit points, paint, re-run image phase, resume. Collects per-frame labels. **Remaining:** the §8 GUI bug/polish backlog, and making it a clean label-collection instrument. |
| 4.5 | **Predictor model + label-gated accuracy** | ⏳ **Next.** Consumes M4's labels. (a) learned `P(error)` QC detector replacing the hand-tuned thresholds; (b) EM-finetuned SAM / micro_sam (only if measured failure rates justify it); plus the relocated label-gated levers. Heavy research content, see §9. |
| 5 | **Aggregate per neuron → Blender** | ⏳ Per-z union of a neuron's chains; export mask stack (and optionally mesh with correct anisotropy). |

Rationale for the order: the GUI *acts on flags*, so flagging (M2) and a queue to clear (M3) are
its inputs; M4.5 *trains on labels*, so the label-collecting GUI (M4) must come first.

---

## 7. Design decisions & knob rationale

Crisp summary of the choices behind the current `PipelineConfig` defaults, the topics the README
cites. Full measurements, the verbose "now landed" annotations, and **rejected alternatives** are in
[`CHANGELOG.md` old §7](../CHANGELOG.md#old-7) and the A/B log
[old §8](../CHANGELOG.md#old-8).

- **Anchor crop (tier-1), default ON.** Run image mode on a high-res crop around the node
  (`crop_anchor`, `crop_size_tif=1200`, `crop_scale=2`), map the box `_crop→_sam` for the video
  seed. Sharpens the *seed* only; cannot fix downstream propagation drift. `crop_anchor=False` =
  legacy full-frame path (M1 baseline).
- **Per-chain crop (tier-2), default OFF globally, AUTO-ON for flagged chains.** Propagate the
  whole chain inside one `_pcrop` window at `chain_crop_scale` for genuinely higher-res masks. A
  per-chain `image_score` fallback (`chain_crop_min_image_score=0.7`) reverts a poor-crop anchor to
  `_sam` *before* wasting a propagation, over-zoom on low-motion chains is a real failure mode, so
  `chain_crop_min_tif=1024` floors the window. A/B (3 neurons): improved 3, regressed 0, net −10
  queue. The auto second-pass (`batch.py`, `tier2_on_flagged=True`) re-runs only QC-flagged chains.
- **Video seed = box + positive point (`box_pos`), default.** Ablation ranked `box_pos` best;
  `pos_only` worst; `mask_only` lost at scale-8. So the **box is kept for AUTO**, but the **GUI drops
  the box** because there a human paints a high-quality mask (the one regime where the mask seed
  wins). `box-from-radius` is **dead** (CATMAID `radius` is mostly placeholder). The confidence-gated
  **mask-vs-box** seed and the **human-painted anchor** are co-built with the GUI (M4), see §8.
- **`multimask_anchor`, landed, default OFF.** Ask SAM2 for its 3 candidate anchor masks and
  auto-select (node-containment → plausible-area → single-CC → decoder IoU). Near-free. Default-off
  preserves the M1 pixel baseline; flipping it on is a label-gated (M4.5) call.
- **`seed_negatives`, landed, default OFF.** Forward `build_prompts`' neighbour-node negatives to
  the video seed. Chain-dependent (helps concave/cluttered, hurts clean) → targeted lever, not a
  blanket default.
- **`box_margin_frac` (underfill fix), validated, default OFF.** A %-of-bbox box pad. Fixes
  genuine under-filled anchors (RIML c25: queue 4→0) but inert/over-padding elsewhere → a targeted
  retry lever (trigger: high-`noskel` + contained anchor), not universal.
- **`postprocess_mask`, landed, default OFF.** Model-free open→close→largest-CC→fill-holes in
  `_sam`, folded into save (so QC scores the cleaned mask). **Suspected of harming results** (see §8
  Theme C), `keep_largest_cc` is dangerous near merges. Treat as on-probation.
- **QC thresholds.** `qc_area_ratio_bounds`, `qc_temporal_iou_min`, `qc_pred_iou_min`,
  `qc_skeleton_dilation_px`, `qc_triage_min_signals` (default 2 = intervene). The `noskel` signal
  dominates raw flags (~80-90%) but the *intervene* core is dilation-robust. `pred_iou` floor
  calibration and `gate_max_area_frac` (0.4 vs 0.75) are **M4.5 label-gated**, leave at current
  defaults until ground truth exists.
- **EM-finetuned SAM / micro_sam, considered, deferred to M4.5.** Not a localised swap (the core is
  SAM2 *video* propagation; micro_sam finetunes the *image* SAM), still imperfect, and accuracy
  isn't the current bottleneck (§4). micro_sam's *napari plugin* is a build-vs-adopt question for the
  GUI, independent of any model swap. See §9 R6.

---

## 8. Backlog / to-do

The active work, reorganized from the old §7-open / §9 field notes into themes and a recommended
tackle-order. Items tagged **[R#]** are research-method candidates collected in §9, flagged for a
literature/deep-research pass before building. Source pointers (old §-refs) point into
[`CHANGELOG.md`](../CHANGELOG.md) for the original wording.

**Recommended order (big picture).** We're at M4 with a working-but-buggy GUI and an
acknowledged-weak detector. The dependency chain is: a *trustworthy* GUI → *unbiased* labels → a
*trustworthy* detector → accuracy levers worth tuning → scale → aggregate. So:
**A (GUI correctness) → B (GUI as label instrument) → C (learned detection) → D (segmentation
accuracy) → E (scale/infra) → F (aggregation/M5)**, with **G (housekeeping)** done opportunistically
alongside.

### A. GUI correctness, fix before trusting any collected label *(do first)*
A correction that the GUI silently mangles poisons every M4.5 label. These gate everything downstream.
1. **Reverse-resume starts from the wrong end (BUG).** A backward resume appears to start at frame 0
   (far end) instead of the corrected frame nearest the anchor, clobbering the correct side. Trace
   how `PropagationSession.propagate` / `propagate_in_video` honor `start_frame_idx` under
   `reverse=True`. *(old §9.2)*
2. **Painted masks change after re-propagation.** Suspect `postprocess_mask` reshaping the stroke
   (distinct from the fixed `correct_as_cond` revert). Action: try disabling post-proc for painted
   masks and confirm the drift stops. *(old §9.2; ties to C/D below)*
3. **Re-propagation loads the whole chain's frames, not just the needed range (perf).** Window
   `init_state` to the propagation direction from the start frame. Couples with bug #1. *(old §9.2)*
4. **Non-central nodes don't auto-pick-up annotations.** Opening a frame whose node isn't the
   central one shows "no positive nodes", seed the annotation onto the frame. *(old §9.2)*

### B. GUI as a clean label-collection instrument, the M4 → M4.5 bridge
M4.5's entire premise is "M4 collects labels." Make the labels *unbiased* and the tool *usable*.
5. **Verify-everything / data-collection presets.** A mode that walks *every* frame (not just
   flagged) so the label store gets the **random sample of un-flagged frames** the learned detector
   requires, without it the model can only shrink the queue, never catch silent errors. **[R1]**
   *(old §9.1; old §7 "GUI as label engine")*
   - *Chain-level half landed (2026-06):* the picker has a **flagged only / everything** mode
     (`gui.py`, `ReviewQueue.all_chains`/`chain_status`), so any chain on disk is openable for
     proofreading, not just flagged ones. The per-*frame* verify-everything sweep is still open.
6. **Marking vs intervention GUI split.** A *marking* mode (sweep frames ok/bad, label-only) and a
   separate *intervention* mode (shows only the flagged frame, exposes correction tools), removes
   accidental edits while scrubbing and the too-many-buttons confusion. *(old §7, old §9.2)*
7. **Explicit save / "confirm-correct" button.** Doubles as a strong positive label for the label
   engine. *(old §9.2)*
8. **GUI usability fixes:** in-editor (napari) notifications for status messages (currently
   terminal-only); image-contrast control; user-guide pass to minimize buttons. *(old §9.2)*
9. **Direction-limited resume.** Resume one direction only, for the case where the other side is
   already correct (and optionally mark that side confirmed-correct → more labels). Edge case; risks
   clobbering the good side. *(old §9.2)*
10. **Export artifacts after a revision:** MP4 generation is broken (gif works); a corrected chain's
    overlay gif/mp4 is stale and should regenerate after a GUI resume. *(old §9.2)*
10b. **GUI recrop for a too-small tier-2 window.** When a tier-2 crop still clips the cell after the
    generous default pad (`chain_crop_pad_tif=512`, 2026-06) and the node-centred collapse default,
    let the reviewer recrop from the GUI. Plan (see the
    `2026-06-22-tier2-crop-sizing-design.md` spec): thread an `override_crop_window` through
    `run_chain_once` / the anchor phase (skip sizing when set), then the GUI builds a window and
    re-runs through that path. Phase 1: a "grow crop by N `_tif` px" button (grows the current
    `crop_window`, clipped). Phase 2: draw a rectangle on the full-res frame for a re-centred window
    (reuses the box-prompt layer). Re-runs `prepare_chain_crop_frames` + anchor + propagate + save, so
    the library/driver boundary stays intact. *(planned; sizing defaults landed 2026-06)*

### C. Error detection / learned QC, the M4.5 accuracy core
The acknowledged weakest part of the system, and arguably the highest-leverage milestone. The notes
argue detection should be improved *before* further accuracy-lever tuning, since the flag rule is a
noisy yardstick.
11. **Build the learned `P(error)` detector first.** Replace the four hand-tuned `qc_*` thresholds
    with one trained probability knob (logistic / small GBT over the signal vector). Split by
    chain/neuron (adjacent frames leak), guard with a held-out randomly-sampled eval set, exclude
    bad-anchor chains, and **log un-flagged samples** (B5), selection bias is the killer. **[R1]**
    *(old §6 M4.5(a), old §7 "GUI as label engine", old §9.1)*
12. **"Flagging is a bad metric", adopt a real quality measure.** Every A/B to date is scored on
    the flag rule itself. Consider an **ERL-style** metric (error-free traced z-distance, separating
    merge vs split errors). **[R2]** *(old §9.1, old §9.3 FFN)*
13. **Detect earlier / in-loop, not post-hoc downstream.** Errors are detected frames *after* they
    start; sample more of the propagation (anchor + ≥1 frame each direction + random + flagged) and
    consider gating/halting propagation on confidence *inside* the loop. **[R3]** *(old §9.1, old §5
    #4 "QC into the loop")*
14. **More / richer QC signals.** Cheapest near-term win: more metrics → more `intervene`
    corroboration + more model features. Add **image-state features** (mask position, local
    intensity/texture, boundary contrast) beyond shape stats. **[R1]** *(old §9.1)*
15. **Strict-by-default flagging for the first labeled campaign.** Flag aggressively for recall
    (`qc_triage_min_signals=1`, tighter bounds); loosen once the learned detector sets the operating
    point. Clear/re-score the manifest on any threshold change (§5). *(old §7, old §9.1)*
16. **Label-gated threshold calibration** (deferred here from M3.5): `pred_iou` floor (read the
    distribution, set against verdicts); `gate_max_area_frac` 0.4-vs-0.75. *(old §7 "QC thresholds")*

### D. Segmentation accuracy / propagation quality
Mostly research-flavoured and label-gated, do once detection (C) is trustworthy enough to measure them.
17. **Make the anchor gate *act* (automatic failed-anchor re-pick).** `score_anchor` currently only
    *records* a verdict. The open piece is letting a FAILed anchor auto-re-pick (e.g. the next node
    toward the chain centre) and auto-escalate prompts *before* queueing a human, turning a bad
    anchor into one frame's compute instead of a wasted ~300-frame propagation. M4 keeps only the
    *human* fallback; this is the automatic half. *(old §7 "Anchor selection", "Anchor prompt quality")*
18. **Post-processing survey.** Standing suspicion that morphological open/close is *hurting* masks
    at scale-8 (blunt for thin neurites; largest-CC is dangerous near merges). First datum: an
    on-vs-off A/B; then survey thin/branching-aware refinement. **[R7]** *(old §9.2)*
19. **Branching / merging, mask covers only one arm of a merge.** A fundamental propagation-memory
    limitation: SAM2's memory biases it to keep tracking the arm it already saw. Likely needs a seed
    on the B-arm at the merge frame (helped by item 22). Makes largest-CC post-proc actively harmful.
    A "merge frame" is a high-risk role for the detector. **[R4]** *(old §9.2)*
20. **Multi-node-per-layer chains read as separate objects.** When several nodes of the *same*
    neuron on one z-layer are really connected, the GUI should recognize that rather than split them.
    *(old §9.2; related to #19)*
21. **Confidence-gated mask-vs-box video seed + human-painted anchor.** The mask seed wins on a
    *trustworthy* anchor (human-painted / tier-2 `_pcrop` / high `image_score`), else fall back to the
    box. The `add_mask` primitive exists; co-build the auto gate with the GUI's human-anchor path.
    *(old §7 "Video seed: box vs mask")*
22. **User-drawn bounding box + extra prompts into video mode.** A manual lever for hard frames, seed
    the B-arm at a merge, re-bound a drifted frame. A *human override*, distinct from the AUTO seed
    finding. *(old §9.2; serves #19)*
23. **Carried-over crop levers:** the `box_margin_frac` auto-retry trigger (high-`noskel` + contained
    → re-run with frac); the GUI "re-propagate a corrected `_sam` chain as tier-2" path (deferred but
    **live**, becomes primary if the tier-2 fallback rate turns out high); tier-3 per-frame tracked
    crop (speculative). *(old §7, old §8.7)*
    - **Crop sizing from the mask, not the skeleton, LANDED default-off (June 2026, A/B-pending).**
      The tier-2 window was sized from the skeleton-node bbox, which traces the *centerline*, so a
      cell whose membrane bulges past the nodes + `chain_crop_pad_tif` gets **clipped** at the window
      edge (measured: AIAL/chain_00 clips 24/113 frames, ≤166 `_pcrop` px against an edge). With
      `radius` dead there's no per-node extent to pad by. New knob `chain_crop_from_mask` (default
      **off**): when on, `chain_crop_window` grows the window to the **union of the skeleton bbox and
      the bbox of the chain's already-saved `_sam` masks** (`pipeline.mask_union_box_px`, over the
      **non-queued** frames so a drifted/merged flagged mask can't inflate the box; falls back to all
      frames if every frame is queued, and to skeleton-only if no usable `_sam` mask exists / the prior
      masks are themselves `_pcrop`). A strict **superset** of the skeleton window, can only grow to
      contain the cell, never clip worse; `chain_crop_max_px` still caps it (trading resolution for
      coverage, the §4 accuracy call). Natural home is the auto second-pass (`tier2_on_flagged`), where
      the first `_sam` pass already wrote the bbox source. A/B harness: `experiments/ab_crop_from_mask.py`
      (headline metric = border-clip frame count); tests `tests/test_chain_crop_from_mask.py`. **Decide
      default-on after the A/B confirms clip→~0 without regressing the queue.**
      - **Test run (June 2026, AIAL/AIZL, "tier-2 everywhere" mode, `batch.TIER2_ALL`+`chain_crop_from_mask`):**
        works and the masks are very good. Clip **improved 24/113 → 9/113** on AIAL/chain_00. Several
        short chains (AIAL 3/4/6/9) **fell back to `_sam`**, their tier-2 *crop anchor* was poor on the
        crop pass, so the item-b safety reverted them. **Correction to an earlier overstatement:** this is
        NOT "they don't benefit from tier-2", higher resolution would likely help their *masks* (the tif
        has the pixels); they simply never *received* tier-2, so the benefit is **untested** (the fallback
        short-circuited it). And it's not (only) over-zoom: for chain_03/04 the recorded `image_score`
        (0.71/0.81) is the **`_sam` recovery** pass, and those `_sam` anchors are *good*, so the *crop*
        anchor was the poor one, i.e. **higher-res EM made SAM2's anchor prediction worse** (the natural-
        image domain gap, FUTURE_DIRECTIONS §2), not better. The fallback is deliberately conservative
        (`chain_crop_min_image_score=0.7`); lowering it trades safety for coverage on borderline chains.
        For 1-3-frame chains there's also little propagation drift to fix (per-frame mask res still helps
        the mesh), so forcing tier-2 there is low reward anyway.
      - **Fallback observability, FIXED (June 2026).** The `_sam` recovery pass overwrote
        `state.image_score`/`anchor_score`, so the final `state.json` showed the *healthy recovery* and
        the *failing crop-pass* reason was lost (only in the run log), which is what made the diagnosis
        above hard. `run_chain` now captures the crop-pass verdict before recovery into three new
        `ChainState` fields: **`fellback_reason`** ("empty-mask"/"gate(...)"/"score<0.7"),
        **`crop_image_score`**, **`crop_anchor_score`** (serialized; None unless `fell_back_to_sam`). These
        are also clean P(error) features (M4.5). Tests in `tests/test_chain_crop_from_mask.py`.
      - **OPEN, residual first/last-frame clip is first-pass-quality-bound (LOG, don't band-aid).**
        The residual 9 clipped frames on AIAL/chain_00 are all at the **start** (fi=0-8), touching
        Top+Right (fi=0: T=60, R=46 `_pcrop` px), exactly the "starting cell cut at top-right" symptom.
        Root cause: the crop is sized from the **coarse scale-8 first-pass mask**, which **under-covers**
        the thin/entering cell at the sensitive first/last frames; the higher-res tier-2 re-propagation
        then recovers *more* cell there than the first-pass mask (which set the window) knew about, so it
        spills past the crop+pad. The queued frame was fi=87 (not the start), so the non-queued exclusion
        is **not** the cause; and the window is **not** at the image boundary (room to grow), so it's not
        an out-of-bounds-of-original-image case. This is the same "crop fidelity is bounded by first-pass
        mask fidelity" limit the lab flagged (mask collapse/explosion at chain ends). Candidate fixes for
        a later pass (owner has ideas): a larger / %-based leeway pad on the mask-derived crop (cheapest,
        but trades resolution + over-zoom risk → A/B); size the crop from a *higher-res* extent (tier-1
        crop mask, or a one-frame hi-res re-predict at the ends) instead of the scale-8 mask; or extend
        the window extra-generously at the first/last frames specifically.
    - **GUI tier-2 layer-scale bug, FIXED (June 2026).** `gui.py` scaled the mask/skeleton/prompt
      layers by `em.shape[1]/W` (height/width) instead of `em.shape[2]/W` (width/width), so on a
      **non-square** `_pcrop` window the overlay stretched by H/W, nodes off the neurite and the mask
      visibly misaligned (square windows were unaffected, which made it look intermittent). On-disk
      masks were always correct; display-only. One-token fix at `gui.py` `open_chain`.
24. **EM-finetuned SAM / micro_sam.** The model swap, revisited only if measured failure rates
    justify it; plus the build-vs-adopt eval of micro_sam's napari plugin. **[R6]** *(old §7
    "micro_sam", old §9.3)*

### E. Scale & infrastructure
25. **Full-dataset run.** The remaining M3 confidence check (~5,206 chains, est. ~20 hr), produces
    the real flag-rate/queue numbers and discharges the deferred M3 full check. *(old §6 row 3)*
26. **`batch.py` follow-ups:** clean up the two stale `TODO[M3]` markers; replace the cached
    `aggregate_data_pv.csv` `annotate_df` source (batch.py:497) with live CATMAID. *(old §6 row 3)*
27. **Multi-GPU chain sharding + parallel review.** Chains are independent → multi-GPU chain-sharding
    is the clean scale-out (the resume design is already most of a work queue). *Parallel review:*
    run batch + GUI concurrently (background works `pending`, human works `flagged`); needs a
    concurrency-safe manifest (partition ownership + file lock), GUI queue polling, and GPU
    arbitration. Measure (telemetry) before buying hardware. *(old §7)*

### F. Aggregation → Blender (M5)
28. **Per-neuron z-union aggregation** of a neuron's chains into `neuron_mask/`.
29. **Chain merge conflicts:** when two chains disagree on a z-slice, plain union, or overlap/voting?
    *(old §7)*
30. **Blender import format:** raw PNG planes vs a single 3D label volume vs pre-meshed `.obj/.ply`
    (marching cubes + decimation), with correct **anisotropy** (§5). **[R8]** *(old §7)*
31. **Instance-label encoding** (uint16, foreground == obj_id) for multi-object aggregation. *(old §5)*

### G. Housekeeping *(opportunistic)*
32. **Repo tidy.** ✅ **Done (June 2026, branch `repo-reorg`).** The durable library
    (`pipeline.py`, `sam2_utils/`, `batch.py`, `gui.py`, `run_aval.py`, `tests/`, `data/`) is
    separated from scratch via `git mv` (history preserved): A/B harnesses + sweeps + logs/figs →
    `experiments/` (with a README mapping each to old §8); reference notebooks → `notebooks/`;
    shelved `calibration.py`/`.ipynb` → `archive/`. Tests stayed green after every move; no durable
    file moved or behaviour changed; nothing deleted. Next-phase scaffolds (`eval/`, `finetune/`,
    `data/groundtruth/`) added as stub-README-only homes. *(old §9.4)*
    **Needs-decision, resolved (lab review, June 2026):**
    - [x] `experiments/*.log` (~1.2 MB) + `experiments/ab_figs/` (6.2 MB), **kept for now**; revisit
      once the Stage 0 evaluation pipeline exists (`experiments/*.log` git-ignored going forward).
    - [x] `archive/calibration.py` + `.ipynb`, **deleted** (shelved approach superseded).
    - [x] `somethin.txt`, **deleted** (stray traceback). *(Still a real latent bug to fix separately:
      `batch.py` chain-dir `mkdir` breaks on neuron names containing `?`, e.g. `VA2?`.)*
    - [x] `datatest.ipynb`, **deleted** (dead exploratory nb).
    - [x] `make_deck_figures.ipynb` + `figures/`, **archived** → `archive/`.
    - [x] `images/`, **deleted** (sample crops; not the actual source used).
    - [x] `experiments/ab_tier2.py`, **deleted** (superseded by `ab_tier2_wide.py` + §8.8 landing).
33. **Docs reorg.** *(This task, substantially done. PIPELINE_CONTEXT split from PIPELINE_HISTORY
    (committed); FUTURE_DIRECTIONS added; README file-structure section updated to the tidied layout;
    file-by-file keep/archive/delete tagging done (item 32 + the per-dir READMEs). Remaining: an
    optional GUI_GUIDE pass.)*

---

## 9. Research directions, novel-method candidates

The items below are **research questions, not engineering tasks**, places where a newer or more
advanced visual-computing method might do better than the current hand-rolled approach. Per §4
(*don't ship on vibes, ground in outside research*).

> **Researched (June 2026).** These have now had a literature/deep-research pass. The SOTA findings,
> the sources, and a staged proposal live in **[`roadmap.md`](roadmap.md)**, the
> forward-looking companion to this file. This section stays as the compact index; the new file is the
> expansion, framed around the project's own difficulties. R-numbers map to FUTURE_DIRECTIONS §4 as:
> **R1→§4.2 · R2→§4.1 · R3→§4.5/§4.2 · R4→§4.3 · R5→§4.7 · R6→§4.7 · R7→§4.4/§4.6 · R8→§4.8.** The FFN
> claims flagged "verify against the source" below are now **confirmed** (Januszewski et al.,
> Nature Methods 2018). Engineering decisions land in §6/§8 as they're made.

- **R1, Learned segmentation-quality / error estimation.** *Serves: items 5, 11, 14.* Replace
  hand-tuned geometric thresholds with a learned `P(error)`, and/or a **separate model that scores
  mask plausibility** ("does this look like a real neuron cross-section?"). Search areas: segmentation
  quality assessment / mask-quality estimation, predictive uncertainty for segmentation,
  out-of-distribution & anomaly detection for masks. Watch: selection bias, anchor contamination,
  covariate shift, train/test split by chain (all detailed in old §7 "GUI as label engine").

- **R2, Error/quality *metric* design (ERL-style).** *Serves: item 12.* The field measures
  error-free **traced distance** and separates **merge vs split** errors rather than counting flags.
  Lead: Januszewski et al., "Flood-filling networks" (Nature Methods 2018; `google/ffn`), Expected
  Run Length. A better yardstick than review-queue deltas for every future A/B.

- **R3, In-loop degradation / halting & confidence-gated propagation.** *Serves: item 13.* Stop or
  flag propagation *when confidence drops*, rather than post-hoc QC on the saved stack. Lead: FFN
  reportedly gates field-of-view movement on predicted mask probability at the FOV border (verify).
  Search areas: confidence-aware tracking, early-stopping / halting in iterative segmentation.

- **R4, Topology changes in propagation (splits / merges / branches).** *Serves: items 19, 20, 21.*
  The "mask covers only one arm of a merge" failure is a known limit of memory-based video object
  segmentation. Search areas: VOS handling of topology change, cell tracking with division/merge
  events, multi-object association at branch points, multi-seed consensus (run from 2+ seeds, flag
  disagreement, FFN, verify).

- **R5, Re-architecture: per-frame dense segmentation + cross-frame association.** *Serves: a possible
  M4.6+ rebuild.* Segment *everything* per frame, then resolve identity across z by overlap/score, sidesteps propagation drift and the merge failure, at the cost of a hard association problem. Leads:
  **Seg2Track** and tracking-by-segmentation; FFN-style **oversegmentation + agglomeration**.
  "Not fun but MUST consider", park unless propagation accuracy plateaus.

- **R6, Domain-adapted EM segmentation models.** *Serves: item 24.* EM-finetuned SAM / **micro_sam**
  / FFN segment neurites markedly better than vanilla SAM2 on the natural-image domain gap. Scope
  ahead: public EM/cell datasets (**MitoEM**, **LIVECell**) for suitability; refactor cost of swapping
  the predictor; and the open question **does finetuning on still *images* improve *video*
  propagation?** (SAM2's video memory vs the image encoder). Plus micro_sam's napari plugin
  (build-vs-adopt).

- **R7, Thin/branching-structure mask refinement.** *Serves: item 18.* Morphological open/close is a
  blunt instrument at scale-8 and may be net-harmful. Search areas: learned/boundary-aware mask
  refinement, thin-structure-preserving morphology, connectomics-specific cleanup. First confirm
  *no* post-proc beats the current one (cheap on/off A/B) before reaching for a method.

- **R8, Surface reconstruction / meshing with anisotropy (lighter weight).** *Serves: item 30.*
  Marching cubes + decimation is the standard path; flagged only so the anisotropy (2/2/50 nm)
  handling and any newer neural-surface options are considered deliberately, not by default.

---

*Update this doc as decisions land, it's the shared big-picture reference. Move closed history
into [`CHANGELOG.md`](../CHANGELOG.md) rather than letting this file grow append-only
again.*