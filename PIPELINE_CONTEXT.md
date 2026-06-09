# Semi-automatic SAM2 segmentation pipeline — context & architecture

Working notes / north-star doc for evolving the `segmentation-playground` codebase
from a notebook into a semi-automatic, human-in-the-loop segmentation tool.

---

## 1. Goal

Segment ~300 *C. elegans* neurons (a few thousand maximal-linear-chains, MLCs)
from an EM `.tif` stack (~300 z-slices) into per-neuron mask volumes, for export
to **Blender**. Everything runs **locally** on one Windows box with a GPU. Keep
infrastructure simple: filesystem only, no database, no server, no web app.

The pipeline is **semi-automatic**: the machine does the work, the human only
reviews and corrects what the machine flags.

---

## 2. Where we are now

**Milestones 1 and 2 are complete** (see §6). The notebook has been lifted into
`pipeline.py` — phase functions plus a `run_chain` driver and `ChainState`
serialization — driven by a thin `run_aval.py` bootstrap. `run_chain` reproduces
the notebook's AVAL masks pixel-for-pixel (verified by diff on the AVAL chain),
with `ChainState` persisted to `state.json`. The notebook below remains the
reference for *what* each phase does; `pipeline.py` is now the source of truth for
*how* it runs.

**M2 (inline QC + flagging) landed.** `run_chain` now has a 9th phase, `run_qc`,
that runs `qc.compute_metrics` over the just-saved chain, writes `qc.csv`,
populates `ChainState.qc_summary` / `triage_frames`, and sets the chain's
`status` to `done` / `flagged` — all headless. QC thresholds are exposed as
`qc_*` knobs on `PipelineConfig` (one place to tune; see §7). The first AVAL run
flags ~38% of frames with 5 `intervene` (≥2-signal) frames — a sane starting
point pending threshold tuning. See §5 for the bugs this surfaced.

**Frame-prep reuse landed (M3 support).** `prepare_video_frames` no longer
re-decodes per chain. It keeps a shared decode cache at
`frames_root/frames_cache_s{scale}/z{file_z}.jpg` (each EM frame downscaled once
ever) and builds a per-chain 0-indexed *link view* at
`frames_root/chain_views/{neuron}_chain{idx:02d}_s{scale}/` (symlink → hard-link
→ copy fallback; on Windows the hard-link branch is the usual path, and works
because cache and views share one volume). Overlapping chains now pay the ~9k×9k
imread+resize once across the dataset instead of once per chain — the prep
bottleneck. Views are namespaced by neuron+chain so a multi-neuron batch can't
collide, and are rebuilt fresh each call. Cached JPEG bytes are byte-identical to
the old per-range writer, so AVAL still reproduces pixel-for-pixel; `run_chain`'s
external signature is unchanged. (Old `sam2_video_{start}_{end}_s{scale}/` folders
from prior runs are now orphaned — safe to delete.)

**M3 batch runner — in place, with reset / scope / telemetry knobs.** `batch.py` is
the headless driver: build the session once, enumerate chains from `chains.json`, run
each through `run_chain`, record status to `output/_manifest.csv` (crash-safe atomic
rewrite per chain; a `running` breadcrumb so an interrupted chain is retried next
launch), and roll per-chain flags up into `output/_triage.csv`. Resume policy is in
`_should_run` (done/flagged skip; failed retries by default; pending/running run). Two
run knobs are surfaced on `run_batch` / `main`: **`neurons`** — an allow-list (e.g.
`["AVAL", "AVAR"]`) scoping a partial run, `None` = all; and **`clean`** — wipe prior
outputs and start fresh. `clean` is scope-aware: with `neurons=None` it removes the
whole `output_root` (full reset); with a subset it deletes only those neurons' chain
dirs and prunes their rows from `_manifest.csv` / `_timing.csv`, leaving other neurons'
finished work intact. `clean` differs from `force`: `force` re-runs in place and lets
`save_masks` overwrite (but `save_masks` never clears `masks/`, so a chain whose frame
coverage shrank leaves orphan PNGs that QC then re-scores) — `clean` deletes first, so
QC only ever scores the current run. Per-chain overlay gifs via `gif_mode`
(`off` / `flagged` / `all`). Per-chain timing + peak VRAM are written to
`output/_timing.csv` (see §7, runtime telemetry).

**Repo hygiene pass (June 2026, pre-GUI).** A lean-up before M4. Coordinate
transforms were fully centralized into `alignment.py` (see §4). Dead code removed:
`sam2_utils/diag_utils.py` (superseded by `diagnostics.py`) and the orphan
`single_object_depth_segmentation.py` script (superseded by `pipeline.py`; the
`_.ipynb` reference notebook stays). Data + output paths now have one home in
`sam2_utils/config.py` (`CSV_PATH` / `CHAINS_PATH` / `ROOTS_PATH` / `OUTPUT_ROOT` /
`FRAMES_ROOT`); `run_aval.py` and `batch.py` import them instead of re-declaring
absolute paths. `diagnostics.py` no longer imports torch at module top (lazy), so
`import sam2_utils` works on a torch-free box. `ChainState.phase_seconds` /
`phase_subseconds` are now declared fields and serialize into `state.json` (were
stamped-on attributes, lost on resume). `batch.py`'s `gif_mode` now honors
`off`/`flagged`/`all` correctly and writes under the passed `output_root`. A
torch-free test suite (`tests/test_alignment.py`, 13 cases) guards the transform
math: run `py -3 tests/test_alignment.py` (or pytest). The only behavioural change
in all of this is the new loud `run_qc` guard; everything else is structure, so a
fresh `batch.py` run reproduces as before.

**M4 (napari review/triage GUI) — core landed (June 2026).** The fourth thin
driver, `gui.py`, plus two torch-free helpers it owns: `sam2_utils/review_queue.py`
(the work queue + a GUI-owned `_review.csv` disposition ledger, kept separate from
the batch's `_manifest.csv`) and `sam2_utils/labels.py` (the per-frame **label
engine** — `_labels.csv`, the M4.5 training data). The GUI reads the batch's flagged
chains, lets a human scrub to flagged frames, edit positive/negative prompt points,
paint an anchor mask, re-run the image phase, and resume propagation over
`PropagationSession`, then rewrites `masks/` + `qc.csv` + `state.json` so a corrected
chain is byte-indistinguishable on disk from a fresh batch run. GPU is lazy (browse/
label needs no predictors). The interactive `PromptRefiner` / `PointClicker`
matplotlib-widget prototypes (below) are now **superseded** by this — they were the
sketch; `gui.py` is the napari rebuild they called for. What's left for M4.5 is
*training* on the labels M4 collects, plus the label-gated accuracy levers; see §6
row 4 for the shipped-vs-deferred split and §7 for the deferred items.

**M4 review-testing pass (June 2026).** Fixes from first real use of the GUI:
(1) **Directional resume** — a correction now re-propagates *away from the anchor only*
(anchor → both ways; a frame after/before the anchor → forward/reverse only), so an
already-corrected center frame is never clobbered when you fix a later one. (2) **Mask
seed, box dropped** — GUI corrections seed propagation with the mask (`add_mask`: the
re-predicted and/or hand-painted mask on the frame), never a derived box; this is the
§7 *box-vs-mask* "human-painted mask is the maximally-verified seed" path, now the GUI
default (the box overlay was removed). (3) **Queue cycling** — prev/next CHAIN cycle
through every undisposed chain *including `in_review`* and wrap, fixing "can't return to
an unfinished chain" (opening a chain marks it `in_review`, which the old next-button
excluded → false "queue empty"). (4) **Painted corrections persist across resumes** —
the GUI video predictor is now built with SAM2's `add_all_frames_to_correct_as_cond=True`
(`setup.build_predictor(kind="video", correct_as_cond=True)`). Without it, painting/clicking
a frame the session had *already tracked* (the normal paint→resume→inspect→repaint loop)
demoted the correction to a *non-conditioning* frame, so the next `propagate_in_video`
re-inferred that frame from memory with `mask_inputs=None` and **silently reverted the
paint** to its pre-correction state. (The *first* correction on a freshly opened chain was
unaffected — an untracked frame is an *initial* conditioning frame regardless of the flag —
which is why it surfaced only on iterative re-touches.) The flag is the SAM2-documented
mechanism ("a frame that receives a correction click becomes a conditioning frame"; it's
`True` in Meta's MOSE-finetune config) and is **inert for the headless batch** — `propagate`
only ever seeds the anchor as an *initial* conditioning frame and never corrects a tracked
frame — so the M1 AVAL pixel-for-pixel reproduction is unchanged and the flag stays
default-off there. This makes the §7 *box-vs-mask* "human-painted mask is the
maximally-verified seed" guarantee actually hold across a multi-correction session. Three
larger items were **documented, not built** (at
the lab's discretion, before/around M4.5): a **marking/intervention GUI split** (sweep
ok/bad in a marking mode, fix only flagged frames in an intervention mode — the
too-many-buttons + scroll-confusion fix), **strict-by-default flagging** (flag
aggressively now for recall, loosen once the M4.5 detector can set the operating point),
and **higher-res masks** (the "pixel-art" resolution complaint — the real fix is the
M4.5 tier-2 per-chain crop; `--hires-em` is the interim EM-only sharpening). See §7.

The notebook `single_object_depth_segmentation_.ipynb` does one chain
end-to-end:

1. Pull CATMAID annotations, apply the stack→tif affine (`alignment.catmaid_to_tif`).
2. Decompose a neuron into MLC chains; pick one chain, take its **mid frame** as anchor.
3. **Image mode** on the anchor: positive skeleton node + K nearest neighbors as
   negatives → mask → largest connected component → bounding box.
4. **Video mode**: seed the box+point on the anchor frame, propagate **bidirectionally**.
5. Save per-frame PNGs.

Reusable pieces already in `sam2_utils/`:

- `config` / `setup` / `catmaid` / `alignment` — stable, keep as-is.
- `viz`, `video_viz` — static + animation display (notebook-oriented).
- **`qc`** — computes the failure signals (see §5) and the composite flag rule;
  thresholds are now parameters. Wired into `run_chain` via `pipeline.run_qc`
  (M2). This was the half-built core of the auto-detection milestone; it is now
  load-bearing.
- **`review`** — read-only proofreading viewer (added alongside M2). Rebuilds the
  overlay from a finished chain's on-disk artifacts (`masks/`, `state.json`,
  `qc.csv`) and delegates rendering to `video_viz`; `grid_flagged` / `animate_flagged`
  show only the QC-flagged frames. Strictly read-only by design — it is NOT the
  M4 intervention GUI (one correction tool, §4). Reuses `qc._iter_mask_paths` /
  `qc._load_binary` so mask reading has a single definition.
- `diagnostics` — VRAM/RAM/disk snapshots for long GPU runs.

The interactive `PromptRefiner` / `PointClicker` classes in the notebook are
matplotlib-widget based and currently **unresponsive in Jupyter**. They are the
prototype for the refinement GUI but should be rebuilt in napari, not patched.

---

## 3. Target architecture

**Not** "notebook → script." The real move is **notebook → library + thin driver**.

### 3a. Library of phase functions
Extract each pipeline step into a small, testable function/class:
`load_frame`, `build_prompts`, `image_predict`, `box_from_mask`, `propagate`,
`postprocess_mask` (candidate, §7), `qc_frame`, `save_masks`, `aggregate`. These
are pure-ish and reused by every driver.

### 3b. Per-chain state machine
An orchestrator carries a **serializable per-chain state object** and decides
which phase runs next:

```
ChainState = {
    neuron, chain_idx, status,            # pending / running / done / flagged / failed
    anchor_frame_idx,
    prompts: {points, labels, box},
    image_mask_ref,                       # path or handle
    qc_summary,                           # flag counts, worst frames
    triage_frames: [...],                 # frames needing human review
}
```

Re-entry (jump back to image mode after a video-mode failure) is just a
transition in this machine. **Serialize state to disk** so a chain can be paused,
resumed, or re-opened in the GUI without recomputation. (Generalizes the
`%store` annotation caching already used.)

### 3c. Interruptible propagation
`propagate_in_video` is a **generator**. Restructure the current two-for-loop
drain so the loop can break at a degrading frame, inject a correction via
`add_new_points_or_box`, and resume.

**LANDED (June 2026) as `pipeline.PropagationSession`.** The monolithic two-loop drain is
now a session object that owns the live `inference_state` and exposes `seed()` /
`propagate(reverse=, start_frame_idx=, max_frames=)` (a lazy generator of `FrameResult`
you can `break`) / `add_points()` / `add_mask()` / `close()`. Continuity lives in
`inference_state`, not the generator: break at a degrading frame, `add_points`/`add_mask`
at it, then `propagate(start_frame_idx=f)` resumes over the *same* mutated state — never
`reset_state` to resume (it wipes prompts; it's the once-per-chain fresh-start call). A
re-propagation overwrites the stale frames it revisits (last write wins). No GUI/napari/torch
import lives in the session — it's the §3c primitive the auto-intervention loop and the M4
GUI both drive. The headless `propagate()` function is now a thin straight-through wrapper
over the session (forward then reverse), so `run_chain` and AVAL reproduction are unchanged.
*Validated torch-free against a fake predictor: per-frame pred_iou capture, break→correct→resume
(`[0,1,2]` then resume `[2,3,4,5]`), and graceful NaN degradation when `_track_step` is absent.*

### 3d. Thin drivers (all call the same library)
- **headless batch runner** — primary mode; runs all chains, writes a triage queue.
- **notebook** — exploration / debugging.
- **napari GUI** — review & correction of the triage queue.

### 3e. Storage layout (filesystem, indexed)
```
output/
  _manifest.csv               # every chain × status — drives batch + resume
  _triage.csv                 # flagged frames across all chains — feeds the GUI
  <neuron>/
    chain_00/
      state.json              # the ChainState above
      qc.csv                  # per-frame metrics
      masks/mask_<z:04d>.png  # canonical space (see §5)
    chain_01/ ...
    neuron_mask/mask_<z:04d>.png   # aggregated per-z union (final, → Blender)

frames_root/                       # SAM2 JPEG frames — separate tree from output/
  frames_cache_s<scale>/
    z<file_z>.jpg                  # shared decode cache; each frame written once
  chain_views/
    <neuron>_chain<idx>_s<scale>/
      00000.jpg ...                # 0-indexed links into cache (init_state reads these)
```

---

## 4. Cross-cutting principles

- **Automation-first, triage-second.** With thousands of chains, auto-run + QC is
  the only viable path; the human is a scarce resource spent only on flagged frames.
- **Supervision-tolerant; the win is propagation, not anchor perfection.** Even at
  one human-approved anchor per chain, the pipeline is ~50× faster than the
  conventional method (hand-painting cells slice by slice). So segmentation-accuracy
  gains — better prompts, finetuned models — are *optimizations*, not requirements:
  judged by whether they shrink the triage queue enough to justify their complexity,
  not by chasing 100% auto-correctness. The expensive thing being automated is the
  ~300-frame propagation, not the one-time anchor. **Confirmed with the lab
  supervisor (June 2026):** full automation is *not* expected given the roughness of
  the data — accuracy with a human in the loop is the goal, and success is simply
  being faster than the old way (hand-colouring cells). This makes accuracy the
  priority over automation-%, which reorders §7: cropping and human-anchor fallbacks
  (accuracy) rank above raw speed. (Frames §7's anchor-quality, cropping,
  human-anchor, and micro_sam items.)
- **One triage queue, one review tool.** Anchor-mask review (image phase) and
  mid-propagation degradation (video phase) are the *same* problem — a flagged
  frame that needs prompt edits. Don't build two GUIs.
- **Checkpoint everything per chain.** Resume on crash; never recompute a finished
  chain. Critical at this scale.
- **Centralize coordinate transforms.** Full-res (`_tif`) / SCALE (`_sam`) /
  SAVE_DOWNSCALE / CATMAID-px (`_cm`) / CATMAID-nm / file-z vs CATMAID-z is the most
  bug-prone area. Put every transform in `alignment.py` and tag variable names with
  their space (e.g. `xy_sam`, `xy_tif`, `xy_cm`). **As of the June 2026 cleanup this
  is enforced, not just aspirational:** the two transforms that were still inline
  bare arithmetic now live in `alignment.py` as `tif_to_sam` / `sam_to_tif`
  (`_tif <-> _sam`, the `/ scale` and `* scale` that were copied across
  `build_prompts`, `anchor_crop_predict`, and `qc`) and `catmaid_z_to_file_z` /
  `file_z_to_catmaid_z` (the `± FILE_Z_OFFSET` reconstructed in `load_frame_sam`
  and `prepare_video_frames`); `nm_to_stack_px` (CATMAID nm -> stack px) joined them
  from `catmaid.py`. So `catmaid_to_tif`, the resolution maps, the z maps, the nm
  divide, and `CropWindow` are now all in one module. A related latent bug is closed:
  `pipeline.save_masks` writes masks at `_sam` (`scale`) and never resamples, but
  `qc` located skeleton nodes by dividing `_tif` by `save_downscale` — silently wrong
  the moment `scale != save_downscale`. `run_qc` now hard-guards `scale == save_downscale`
  (passes under the canonical config) so that divergence fails loudly instead of
  producing wrong QC. **As of the M3.5 crop (June 2026)
  there is a fourth space, `_crop`** — the high-res anchor crop — and
  `alignment.CropWindow` is its single home: `_crop ↔ _tif ↔ _sam` mapping happens
  there and nowhere else, with the only row/col (`[y,x]` vs `(x,y)`) swap isolated to
  `CropWindow.slice_tif`. This **narrows what `scale`/`_sam` mean**: `_sam` is now the
  *video*-propagation input space **and** the canonical on-disk mask space — no longer
  "everything the predictors see," because the image phase runs in `_crop` and only its
  *box* is mapped back to `_sam` for the video seed. `scale` is **not deprecated**; it
  is the bridge that lands the crop result in propagation space, and still defines the
  mask space (`save_downscale == scale`). It only recedes if tier-2 per-chain cropping
  ever moves propagation itself off the scale-8 full frame. Corollary the crop forces:
  any gate tolerance expressed in `_sam` px (contain radius, box margin) is rescaled by
  `scale/crop_scale` when scoring in `_crop`, and `area_frac` becomes crop-relative
  (see §7 *QC thresholds*).
- **Keep it simple.** Local, single box, filesystem. No premature distributed
  anything.

---

## 5. Known issues to resolve in the refactor

These currently break or will break the QC step — fix them when extracting the library.

**Mask-space / filename issues (§5.1, §5.2) — RESOLVED in milestone 1.**
Canonical on-disk mask space is now fixed in one place (`PipelineConfig`):
masks are stored at `_sam` space with `save_downscale == scale == 8`, so there is
no resample and no 2× skeleton-containment offset. Files are named
`mask_<catmaid_z:04d>.png` (no `z` prefix), which `qc._iter_mask_paths` parses,
and are written as **0/255 uint8 single-channel** (the notebook's format) by
`pipeline.save_masks` — directly viewable and pixel-comparable to the notebook.
`qc._load_binary` thresholds `> 0`, so it reads them unchanged. Note: this means
`pipeline.save_masks` does **not** use `qc.save_masks`, which writes uint16
*instance labels* (foreground pixel == obj_id). For a single object obj_id is 1,
and value-1 in a 16-bit image looks empty and is destroyed by any 16→8-bit
conversion — that was the original "empty masks" red herring. Instance-label
encoding is a multi-object concern, deferred to M5 (see §7).

1. ~~**Filename convention mismatch.**~~ Resolved — see above. (`mask_<catmaid_z:04d>.png`.)
2. ~~**Mask-space mismatch.**~~ Resolved — see above. (`save_downscale == scale`, no resample.)
3. **`pred_iou` — RESOLVED (populated, June 2026).** SAM2 *computes* the mask-decoder
   IoU head (`ious`) but `track_step` discards it before it reaches `inference_state` or
   the `propagate_in_video` yield (trace: `_forward_sam_heads` → `track_step` in
   `sam2/modeling/sam2_base.py`; the value is unpacked to `_`). It is still in hand one
   level down, in `_track_step`'s return (`sam_outputs[2]`). `pipeline._attach_iou_hook`
   wraps `_track_step` **read-only** to record `float(ious.max())` per frame (max == the
   argmax SAM2 itself selects on a multimask anchor; the lone value when single-mask).
   `propagate()` now returns `(video_segments, frame_conf, pred_iou)`; `run_chain` maps
   frame_idx→z and hands it to `run_qc`, which forwards it to `qc.compute_metrics(pred_iou=…)`.
   The hook is **best-effort**: if a future SAM2 refactor changes `_track_step`, pred_iou
   falls back to NaN (the flag rule already treats NaN as inert) rather than crashing.
   **Consequence:** the 4th flag signal is now live — `flag_count` can reach 4, and the
   "stable-but-wrong" frames the §7 item-4 *Bound* called out (plausible area, good
   temporal overlap, node inside mask) can finally trip a signal. `frame_conf`/`logit_conf`
   (per-frame mean-foreground-sigmoid confidence, recorded in `qc.csv`) stays as a secondary
   diagnostic, no longer the only confidence proxy. *(Earlier state, for the record: at M2
   `propagate` already stopped throwing the logits away and returned `frame_conf`, but the
   calibrated decoder IoU itself was still discarded — that is the part now resolved.)*
4. **QC post-hoc → in-run (M2).** `run_chain` now calls `pipeline.run_qc` right
   after `save_masks`, so QC + flagging happen as part of every run, headless.
   It still reads the just-written PNGs back rather than scoring *inside* the
   propagate loop — that fully-interleaved form is only needed for
   *halt-and-re-prompt*. The §3c generator restructure it depends on has now **landed**
   (`PropagationSession`), so the mechanism exists; wiring QC *into* the loop as an
   auto-intervention policy is the remaining **M4** step. So: "QC moved into the run,"
   not yet "into the loop."
5. **Anisotropy for Blender.** Voxels are 2/2/50 nm; at SCALE=8 that's ~16/16/50 nm
   (z ≈ 3× xy). Whatever meshes the volume must receive correct z spacing or the
   neuron will be squashed in z. *(Still open; M5.)*
6. **Per-chain skeleton for containment (M2, AVAL-surfaced).** The
   `skeleton_contained` probe must use *this chain's* skeleton nodes, NOT the whole
   neuron's. First AVAL run flagged 100% of frames because `compute_metrics` was
   handed the full neuron (`skeleton=annotate_df, cell_name="AVAL"`); AVAL is ~24
   chains, so its nodes cross each z at several xy and their centroid sits off any
   single process → never inside the mask, even at the anchor. Fix: `run_chain`
   filters `annotate_df` to `chain["nodes"]` before QC. Also, `skeleton_contained`
   is now **tri-state** — `True` / `False` / `NaN` (no chain node at that z; a
   non-monotonic neurite leaves the section) — and only an explicit `False` flags,
   so the ~30% of frames in z-gaps abstain rather than false-flag. This is a
   concrete instance of the §4 "centralize coordinate transforms / tag the space"
   principle: the bug was a *reference-set* error (whole-neuron vs chain), invisible
   until eyeballed on AVAL.
7. **Manifest is append-mode; a scoring-threshold change mid-campaign silently mixes
   thresholds (M3.5, June 2026).** `_manifest.csv` rows are written/kept per chain as
   they run, not rewritten. If a gate threshold changes between runs (e.g.
   `gate_max_area_frac` 0.05 → 0.4), chains scored earlier keep their old-ceiling
   `anchor_passed`/`anchor_reasons` while new chains use the new ceiling — so
   `anchor_passed` silently reflects two configs at once. *Symptom:* `area`-FAIL rows
   whose `anchor_area_frac` is *below* the supposed ceiling (seen pre-clear: area-FAILs
   down to 0.055 alongside PASS rows up to 0.38). *Fix:* clear the manifest (or force
   re-score) after any scoring-threshold change; sanity-check uniformity by confirming
   min(`area_frac` among area-FAILs) > max(`area_frac` among PASS).

The three failure signals you listed already exist in `qc`:
`area_ratio` (size change), `temporal_iou` (overlap), `skeleton_contained` (node
containment), plus a composite flag/intervene rule. Auto-detection is mostly
**moving these inline** + tuning thresholds, not building from scratch.

---

## 6. Milestone roadmap (refined)

Mapped to your original numbering in brackets. Reordered so each step's input
exists before it's built.

| # | Milestone | Notes |
|---|-----------|-------|
| 1 | ✅ **Refactor to library + state machine + serialization** [your 1] | **DONE.** Phase functions + `run_chain` + `ChainState`/`state.json` in `pipeline.py`; `run_aval.py` bootstrap; reproduces AVAL masks pixel-for-pixel. Interruptible propagation **landed** as `PropagationSession` (§3c) — the M4 "task zero" core-library change is in; what remains for M4 is the GUI/auto-intervention *policy* on top, not the mechanism. |
| 2 | ✅ **Inline QC + flagging** [your 2 + 5] | **DONE (QC in-run).** `run_chain` step 9 = `pipeline.run_qc`: `qc.compute_metrics` over the saved chain → `qc.csv` + `ChainState.qc_summary`/`triage_frames` + `done`/`flagged` status. Thresholds exposed as `qc_*` on `PipelineConfig`. Validated on AVAL: caught the whole-neuron-skeleton bug (§5 #6), now ~38% flagged / 5 intervene. Caveats: fully-inline-in-loop QC (§5 #4) still deferred to M3/M4, and calibrated `pred_iou` was deferred at M2 (since landed — §5 #3). Read-only `review` viewer added for proofreading. *Indexed storage (`_manifest.csv` / `_triage.csv` across chains) belongs to M3.* |
| 3 | ✅ **Headless batch runner + resume** [your 6] | **DONE (validated on subset, June 2026).** `batch.py`: manifest ledger, run all chains unattended, `_triage.csv`, `clean`/`neurons` knobs, runtime telemetry (§7). Validated end-to-end on a 24-neuron / 384-chain / 9.4k-frame subset run: `_manifest.csv` + resume + `_triage.csv` + `_timing.csv` all produced, 0 errors, compute-bound (~1.7 hr wall ≈ 1.55 hr compute). The `run_chain` arg list is confirmed by the successful run; the two remaining `TODO[M3]` markers in `batch.py` are now stale and should be cleaned up (the cached-CSV-vs-live-CATMAID `annotate_df` source at `batch.py:497` is the one real follow-up — currently relies on the cached `aggregate_data_pv.csv`). A *full*-dataset run (5,206 chains, est. ~20 hr) is the remaining confidence check, not a blocker. Now lets us measure the auto-flag rate *before* investing in the GUI — see §7. |
| 3.5 | **Headless anchor-quality hardening (pre-GUI)** | Every change that shrinks and de-contaminates the triage queue *before* the GUI sees it — all headless. Motivated by the M3 result (~90% of flags are `noskel` on otherwise-healthy masks) and to clean the labels M4 will collect. **Internal order, measure-gated (§4):** (0) **automatic threshold-sensitivity sweep** over `qc_skeleton_dilation_px` (sensitivity, not correctness; gold-set labeling shelved, §7); (1) **anchor-quality gate** before propagation (score: containment / single-CC / plausible area) — **scoring half LANDED (June 2026), OBSERVATIONAL**: `pipeline.score_anchor` records a per-chain verdict to `ChainState.anchor_score` and to the new `_manifest.csv` `anchor_passed/anchor_reasons/anchor_contained/anchor_lcc/anchor_area_frac` columns (joins to `_triage.csv` on `neuron,chain_idx`); it does **not** branch yet. The automatic failed-anchor re-pick (`# [M4]` marker in `select_anchor`/the empty-mask path) still moves here; M4 keeps only the *human* fallback. (2) **anchor treatments — now applied by DEFAULT, not gated on a FAIL** (decision June 2026, supervisor-authorized; **reverses** the original "escalate-actions hanging off the gate" framing. Rationale: §4 accuracy-first, plus the AVAL result that a scale-8 gate never fires on clean-but-coarse thin anchors — gating the crop behind it would skip exactly the cases that need resolution): **anchor full-res crop LANDED as the default image phase** (`crop_anchor=True`, `crop_size_tif=1200`, `crop_scale=2`; runs image mode in `_crop` via `alignment.CropWindow`, maps the box `_crop→_sam`; `crop_anchor=False` keeps the legacy scale-8 path for A/B + the M1 regression baseline). **`multimask_output` auto-select LANDED (June 2026), default-off** (`multimask_anchor=False`; `pipeline._select_anchor_mask` ranks SAM2's 3 candidates by node-containment → plausible-area → single-CC → decoder IoU and feeds the winner's box to the seed — near-free, since the decoder computes all 3 either way and `set_image` runs once; **default-off** for two reasons — it changes which anchor mask is chosen so it would break the M1 pixel-for-pixel regression baseline, and per the §6 ruler "free to *run*" ≠ "shown to shrink the queue," so flipping it **on by default** is itself a label-gated call (M4.5). For now it ships as an A/B switch carried into M4's evaluation). **`box-from-`radius`` is DEAD** — the CATMAID `radius` column is mostly placeholder values, so a box synthesized from it is confident-but-wrong extent, worse than a bare point (decided June 2026). Still open: **confidence-gated mask-vs-box video seed — deferred to M4, co-built with the GUI** (its maximally-verified case is the *human-painted* anchor, an M4 mask-edit feature, and evaluating "when does the mask seed beat the box" needs M4 labels; the `add_mask` primitive already exists on `PropagationSession`, so the auto/confidence-gated path lands alongside the GUI's human-anchor path — see §7 *Video seed: box vs mask*). (3) **auto neighbour-node negatives** at the video seed — **LANDED (June 2026), default-off** (`propagate(..., seed_negatives=False)`; forwards `build_prompts`' same-z neighbour negatives to the video seed when on; §7 *Negative points*). (4) **chain-level verdict gating** on intervene frames. (5) **mask post-processing** — **LANDED (June 2026), default-off** (`pipeline.postprocess_mask`: open→close→largest-CC→fill-holes in `_sam`, folded into save so it runs *before* QC; §7 *Mask post-processing*). **AVAL crop A/B (June 2026, same 24 chains, crop on vs off):** containment held 24/24, `anchor_lcc` median 0.996; the new anchor FAILs are **threshold artifacts** (`area` from the now-crop-relative `area_frac` ceiling at the old `0.05`), not real misses; downstream queue ~unchanged (96→94% `noskel`) — tier-1 crop sharpens the **seed** but cannot fix downstream propagation drift, and AVAL (thick) is the wrong neuron to show crop's payoff. **Clean 5-neuron run (June 2026 — AI* group: AIAL/AIAR/AIYL/AIZL/AIZR, 233 chains, uniform `gate_max_area_frac=0.4` after the §5#7 manifest-clear):** anchor gate **230/233 pass (98.7%)**, **containment 233/233**, area limb clean (1 legit-large `area` FAIL, 2 borderline `frag` that propagated fine); the triage queue stays **~82% `noskel`** (72% noskel-only; 80/572 flagged frames `intervene`) — confirming on clean thin-neuron data that crop + items 3/5 sharpen the seed but do not touch propagation-resolution drift. **Still not the closing run** (5 of 16 neurons); the full 629 — plus a controlled items-3/5 A/B (same chains, flags toggled) and the `gate_max_area_frac` 0.4-vs-0.75 call — remain, and the full run discharges M3's full-dataset check. **Next levers:** item 0 **answered** — the `qc_skeleton_dilation_px` sweep (0..10 px) showed the *intervene*-level queue is dilation-robust (`intervene_rate` moved <0.005 across the sweep, d=3 → `n_intervene=80`), so the multi-signal queue does not hinge on the `noskel` threshold; the bulk single-signal `noskel` stays ambiguous (real propagation miss vs. benign branch / z-gap centroid), **pending M4 labels**. item 4 ✅ **landed (§7): triage queue gated on `intervene`** (`_triage.csv` 572→80 on the 5-neuron run, chain statuses unchanged). **The remaining accuracy levers are now label-gated and have moved out of M3.5 → M4.5** (decision June 2026): the tier-2 per-chain crop for the *genuinely-real* `noskel`, the `pred_iou` floor calibration, and the `gate_max_area_frac` 0.4-vs-0.75 finalization all need M4's ground-truth labels to know the `noskel` residue is real drift (not a benign branch / z-gap centroid) before they can be tuned or trusted — so they belong after the GUI, in the predictor-model milestone, not in pre-GUI M3.5. So M3.5 is **complete as a pre-GUI headless phase**; what's left of its agenda is either an M4 GUI co-build (mask-vs-box seed) or M4.5 label-gated work. Next: **the full-dataset run** (629 chains / the full 5,206 — discharges M3's deferred full check and is the only remaining M3.5-era item), then **M4**. **Ruler (gold set scrapped):** relative deltas at fixed thresholds — anchor-gate pass rate (lean on `lcc`/area sanity) + queue/flag-distribution change; cannot certify accuracy or surface silent errors (→ M4 labels). |
| 4 | **napari review/triage GUI** [your 3 + 4] | ✅ **CORE LANDED (June 2026)** as `gui.py` + `sam2_utils/review_queue.py` + `sam2_utils/labels.py` (build path chosen over adopting micro_sam's plugin — see DEFERRED at end of this cell). One tool, read-from-the-queue: open a flagged chain, scrub to flagged frames, add/remove points (incl. **negative** points, §7 *Negative points*), re-run image phase, resume propagation (over `PropagationSession`, the interruptible primitive — ✅ **landed and AVAL-validated bit-identical**, §3c; the M4 "task zero" core-library change is done). Covers anchor review *and* mid-propagation. **Architecture as built:** a *thin driver* composing the existing library — `review.load_chain` rebuilds the on-disk overlay, `review_queue.ReviewQueue` is the work queue + a GUI-owned `_review.csv` disposition ledger (separate column from the batch's `_manifest.csv`, the cheap form of the §7 parallel-review "partition ownership"), `labels.LabelStore` is the per-frame label engine (below), and the GPU `PropagationSession` / `image_predict` / `run_qc` calls reuse `pipeline` verbatim. **Two-tier loading:** the light tier (annotate_df + chains + on-disk artifacts) browses/scrubs/inspects/paints/labels with no torch; the SAM2 predictors build lazily only on the first re-run/resume, so a reviewer can work while the background batch holds the GPU (the §7 parallel-review payoff, minus the concurrency lock). A correction rewrites `masks/` + `qc.csv` + `state.json` so the chain is indistinguishable on disk from a fresh batch run. **View conveniences:** auto-zoom to the mask bbox on open / jump-to-flagged, tunable point size (default 4 `_sam` px), and an opt-in full-res EM background (`hires_em`) that reads the original tifs and scales the masks/points to overlay them — sharpening the *EM context* only. **The displayed mask is scale-8** (the `_sam` propagation/storage space), and that is a hard limit of the GUI: the high-res anchor crop sharpens only the seed (its box → `_sam`, crop discarded), so genuinely higher-res masks need the **tier-2 per-chain propagation crop, which is M4.5** — `hires_em` is the cosmetic half (crisp EM under a still-scale-8 mask), the real fix is label-gated. Torch-free pieces unit-tested (`tests/test_labels.py`, `tests/test_review_queue.py`, 24 cases); napari 0.7 layer/widget APIs validated, live Viewer verified interactively (needs a GL context). **DEFERRED to M4.5 / later (marked `# [DEFERRED]` in code):** crop-space anchor re-predict (GUI re-predict uses the legacy full-frame `_sam` path that matches the displayed frame); the *automatic* confidence-gated mask-vs-box seed (human-painted-mask seed path is wired, the auto gate is label-gated); cross-process file lock + live auto-poll + multi-GPU arbitration for *concurrent* reviewers (single-reviewer is safe, `refresh()` is poll-on-demand); and the micro_sam napari-plugin *adopt* spike. **The §7 open items it implements/owns:** the *human-painted anchor* mask-edit surface and human-anchor routing for tiny / failed-anchor chains (§7 *Video seed: box vs mask*); the **human-escalation fallback** when the pre-propagation anchor-quality gate fails after auto-retries (§7 *Anchor selection*, *Anchor prompt quality*); the **build-vs-adopt** evaluation of micro_sam's napari plugin before writing a GUI from scratch (§7 *micro_sam*); and the **parallel-review architecture** — a separate *review*-status manifest column owned by the GUI, queue polling/watching, and GPU arbitration for interactive re-runs (§7 *Performance scaling → parallel review*). **New responsibility (§7 *GUI as label engine*):** every correction is a training label, so M4 logs per-frame rows (features + verdict + anchor verdict + role + rule-flagged) into a label store it owns — including a random sample of *un-flagged* frames. M4 *collects* the labels; **training the model on them is M4.5** (the boundary between the two milestones). |
| 4.5 | **Predictor model + label-gated accuracy** [new, June 2026] | The milestone that **consumes M4's label exhaust** — everything that needs ground truth to build or tune, so it cannot precede the GUI. Two model threads: (a) the learned **`P(error)` QC detector** (§7 *GUI as label engine*) — replace the four hand-tuned `qc_*` thresholds with one trained probability knob, modelled boringly (logistic / small GBT over the signal vector), split by chain/neuron, guarded by a held-out eval set; (b) the **EM-finetuned SAM / micro_sam predictor** (§7 *micro_sam*) — the segmentation-model swap, revisited *only* if measured failure rates justify it. Plus the **label-gated accuracy/threshold levers relocated from M3.5**: the **tier-2 per-chain crop** for genuinely-real `noskel` drift (the lever that actually moves downstream propagation resolution, gated on labels confirming the drift is real); the **`pred_iou` floor calibration** (read the distribution, set the floor against verdicts — it feeds detector (a)); and the **`gate_max_area_frac` 0.4-vs-0.75 finalization**. `pred_iou` is already populated (§5 #3) so it logs from the first M4 frame. NB the napari-plugin *build-vs-adopt* eval of micro_sam stays in M4 (it's a GUI question); only the *model swap* is M4.5. |
| 5 | **Aggregate per neuron → Blender** [your 7] | Per-z union of a neuron's chains; export mask stack (and optionally mesh, with correct anisotropy). |

Rationale for the reorder: the GUI *acts on flags*, so the flagging logic (2) and
a queue to clear (3) are its inputs. Building the GUI first means building it blind.

---

## 7. Open decisions (not blocking)

*Tagged by milestone (see §6): **M3.5** = headless anchor-quality + calibration harness (pre-GUI); **M4** = napari GUI (collects labels); **M4.5** = predictor model + label-gated accuracy (consumes M4's labels — the learned `P(error)` detector, EM-finetuned SAM, and the tier-2 crop / `pred_iou` floor / `gate_max_area_frac` levers that need ground truth to tune); **M5** = aggregation → Blender. "auto · human→M4" means the automatic part lands in M3.5 and the human-interaction part stays in M4; "M4 logs · M4.5 trains" splits label collection (M4) from model training (M4.5). Untagged items are unscheduled.*

- **[M5]** **Blender import format:** raw PNG planes (simplest) vs. a single 3D label
  volume vs. pre-meshed `.obj/.ply` (marching cubes + decimation). Affects §5.5.
- **[M3.5]** **QC thresholds:** defaults (area_ratio ∉ [0.5, 2.0], temporal_iou < 0.3,
  pred_iou < 0.5) are now `qc_*` knobs on `PipelineConfig` — tune there, not in
  `qc.py`. First AVAL run: 40/104 flagged (38%), 5 intervene, with `skel miss: 36`
  (node present, mask doesn't cover it) vs `skel n/a: 31` (no node at that z). The
  36 containment misses are the next thing to eyeball — are they real drift, or
  is `qc_skeleton_dilation_px` (3) too tight for thin neurites? Tune before
  trusting the flag rate at scale.
  **Subset-run result (M3, June 2026, 24 neurons / 260 flagged chains / 3,789
  flagged frames):** `skeleton_contained=False` (`noskel`) drives ~90% of all
  flags, while `area_ratio` (median 0.99) and `temporal_iou` (median 0.67) look
  healthy on flagged frames — so the headline flag rate is currently a
  skeleton-containment artifact, not a degradation signal. Only 16% of flagged
  frames are intervene-level (≥2 signals), so plan around *intervene*, not raw
  flags. **Item-0 sweep (RAN, June 2026):** the automatic `qc_skeleton_dilation_px`
  sweep (0..10 px) over the saved masks showed the *intervene*-level queue is
  dilation-robust — `intervene_rate` moved <0.005 across the sweep — so the multi-signal
  queue does not hinge on the `noskel` threshold. What the sweep *cannot* say is which
  vanishing single-signal flags were real errors; that correctness call waits for
  M4-collected labels (manual gold-set shelved, see *Manual gold-set labeling* below).
  **Update (June 2026): `pred_iou` populated.** pred_iou now comes from SAM2's
  mask-decoder IoU head (§5 #3). Enabling it **changes the flag/queue distribution** — it's
  a real 4th signal at `qc_pred_iou_min` (default 0.5). Per the §5 #7 mixed-threshold
  discipline, **clear/re-score the manifest** on the first run with pred_iou on, or early
  chains (NaN, inert) silently mix with later chains (live). To record pred_iou *without*
  flagging on it (observe first, per the §6 measure-then-trust ruler), set
  `qc_pred_iou_min <= 0`. First task before trusting it at scale: read the pred_iou
  distribution on a clean run and confirm 0.5 is the right floor — it's a borrowed default,
  not yet calibrated on this data. **This calibration is now M4.5** (label-gated): the floor
  is set against ground-truth verdicts, and pred_iou is the strongest feature for the M4.5
  learned detector, so calibrating it belongs in the predictor-model milestone, not pre-GUI.
  Likewise the **`gate_max_area_frac` 0.4-vs-0.75 finalization** (below) moves to M4.5 — it
  needs labels to choose; until then leave it at 0.4 and filter `area`-only FAILs by
  `anchor_contained`/`anchor_lcc`.
  **Crop A/B (M3.5, AVAL, June 2026 — default crop on vs off, same 24 chains):** the
  anchor gate is now scored in `_crop`, which makes its thresholds **space-relative** —
  the concrete fallout of the §4 `scale` narrowing. (a) `area_frac` is measured against
  the crop, not the full frame, so it jumped ~35× (median 0.0004 → 0.014); the old
  `gate_max_area_frac = 0.05` ceiling mis-fires on clean thick AVAL anchors
  (`contained=True`, `lcc≈1.0`), producing spurious `area` FAILs. **Action: raise
  `gate_max_area_frac` for crop space** (the AVAL data put the floor at ≈0.4). (b)
  `gate_min_area_frac = 1e-5` is now **inert** — the small-cell false-positive worry is
  *resolved* by the higher resolution. (c) The containment radius and box margin are
  auto-rescaled by `scale/crop_scale` (×4) inside `run_chain`, so
  `qc_skeleton_dilation_px` stays the one knob and keeps anchor- and per-frame
  containment physically equal. Until the ceiling is set, filter FAILs by
  `anchor_reasons` (an `area`-only FAIL with `anchor_contained=True` and high
  `anchor_lcc` is a clean anchor, not a failure).
  **Clean 5-neuron run (M3.5, June 2026 — 233 chains, AIAL/AIAR/AIYL/AIZL/AIZR, uniform
  `gate_max_area_frac=0.4` after the §5#7 manifest-clear):** the first trustworthy
  single-ceiling thin-neuron read (an earlier 0.05-ceiling partial batch was discarded
  as mixed-threshold, §5#7). (a) **Anchor 230/233 pass (98.7%); containment 233/233;
  median `anchor_lcc` ≥0.995 per neuron** — the seed half is solid across thin neurons.
  (b) **Area limb clean at 0.4:** one `area` FAIL, AIZR 7 (`area_frac` 0.66, `lcc` 0.99,
  contained — a legit-large clean anchor; would pass at 0.75). (c) **The 2 `frag` FAILs
  are borderline false alarms:** AIYL 34 (`lcc` 0.799) and AIAR 8 (`lcc` 0.761) sit at
  the 0.8 `min_largest_cc_frac` and both propagated clean (`flag_rate` 0) — so `frag`
  alone should *flag*, not hard-gate. **Decision still open:** keep 0.4 (accept the rare
  legit-large `area` FAIL) vs raise to 0.75 (0 `area` FAILs, lean on containment +
  `frag`). (d) **The triage queue is still noskel-dominated and unmoved:** of 572
  flagged frames, `noskel` is present in 82% and is the *only* signal in 72%; only 80
  are `intervene` (29/233 chains `flagged`) — the same 80–90% signature as M3 and the
  pre-clear 376-run, so crop + items 3/5 do not touch it, as forecast. The 14%
  intervene-vs-flag split is the item-4 case, now landed (the queue surfaces the 80
  `intervene` frames, not the 572; see item 4 below); whether the 82% single-signal `noskel`
  is real drift or a `dilation_px=3` artifact was the item-0 question — the sweep (item 0,
  RAN) showed the *intervene* core is dilation-robust, but the pure-`noskel` residue stays
  ambiguous pending M4 labels.
- **[M3.5 — superseded by item 4]** **Chain-level verdict:** `qc_intervene_to_flag_chain` (default 1) marked a chain
  `flagged` when ≥1 frame hit `intervene` (≥2 signals); `triage_frames` originally
  listed every single-signal flag. Subset run (M3) confirmed the split matters: raw
  flagged chains (260) vastly outnumber chains with any intervene frame, because a
  single `noskel` flag marks a chain `flagged`. **Resolved by item 4 (below):** both
  the per-frame queue and the chain verdict now key on the same queue definition
  (`flag_count >= qc_triage_min_signals`, default 2 = intervene), so `triage_frames`
  no longer lists single-signal flags.
- **[M3.5 — landed] Item 4 — triage queue gated on `intervene` (June 2026).** Both the
  per-frame human queue and the chain verdict now surface only frames at/above a configured
  severity, `flag_count >= qc_triage_min_signals` (new knob on `PipelineConfig`, default
  **2 = intervene**; set to 1 for the legacy "queue every flag"). `run_qc` writes a `queue`
  column to `qc.csv` (so the cross-chain rollup filters on the artifact alone, §4) and adds
  `n_queue` / `queue_rate` to `qc_summary`; `n_flagged` is **retained** — single-signal flags
  stay on disk as diagnostics and as M4 label fodder, just not surfaced to a human.
  `triage_frames` and `_triage.csv` are now the queue (intervene) frames; chain `status`
  keys on `n_queue` so the two can never disagree. **Behaviour-preserving at defaults:**
  `min_signals=2` → `n_queue == n_intervene`, identical to the prior
  `n_int >= qc_intervene_to_flag_chain` rule. `batch.build_triage_queue` filters
  `queue → intervene → flag` (fallback chain), so a pre-patch run rebuilds straight to the
  intervene set with no re-segmentation.
  **Validation (clean 5-neuron run, 233 chains / 2082 frames):** rebuilding `_triage.csv`
  off the existing `qc.csv` (intervene fallback) drops it **572 → 80 queued frames** —
  exactly the item-0 sweep's d=3 `n_intervene=80` — with `_manifest.csv` statuses unchanged
  at **29 flagged** chains (the verdict was already intervene-gated). The 80 are
  *multi-signal-corroborated*, not `noskel` survivors: e.g. AIAL/0 z1548 = `area×3.9` +
  `tIoU 0.02` (a propagation runaway), and several queue frames have `skeleton_contained=True`,
  flagged purely on `area` + `tIoU`. This is the dilation-robust core item 0 identified —
  `intervene_rate` moved <0.005 across the 0..10px sweep.
  **Bound (→ §5 #3):** with `pred_iou` now populated (§5 #3) `flag_count` can reach 4, so the
  queue gains a 4th corroborator beyond the three *geometric* signals (`area_ratio`,
  `temporal_iou`, `skeleton_contained`). This directly attacks the failure mode this Bound
  originally called out — a stable-but-wrong frame (plausible area, good temporal overlap,
  node happens to fall inside the mask) tripped none of the three geometric signals, could
  not reach `intervene`, and stayed invisible; SAM2's decoder IoU is the signal most likely
  to surface it. (Pre-pred_iou, `flag_count` topped out at 3 and this case was uncatchable —
  that is the gap now closed in mechanism, still to be confirmed at scale once the pred_iou
  floor is calibrated, §7 *QC thresholds*.)
  **`review` left on `flag` by choice** — `grid_flagged` / `animate_flagged` still eyeball
  *all* flags (the "are the flags sane" diagnostic, the item-0 use case); the human *work*
  queue is `_triage.csv`. A one-line switch to `queue` is noted in the patch if that
  distinction ever wants collapsing.
- **[shelved]** **Manual gold-set labeling (`calibration.py`) — dropped for now.**
  The plan was a read-only human labeling pass — the **anchor** as a chain-level
  gate, **all flagged frames** (precision), and a **uniform random sample of
  un-flagged frames** (to estimate the silent-error rate the queue can't show) —
  producing ground truth to calibrate the `qc_*` rule. **Decision (June 2026):
  shelved** — too much manual effort up front. Ground-truth labels are deferred to
  M4's GUI logging (*GUI as label engine*, below); M3.5 measurement falls back to
  automatic proxies (anchor-gate pass rate + queue deltas at fixed thresholds).
  `calibration.py` stays parked in the repo, off the critical path. The two
  structural facts it encoded still hold and still constrain M4's logging:
  (a) anchor quality gates everything downstream — log the anchor verdict and keep
  bad-anchor chains out of training; (b) a detector can only be measured against
  truth, never its own flags — so the GUI *must* still log a random sample of
  un-flagged frames, or the eventual model stays blind to silent errors.
- **[M4 logs · M4.5 trains]** **GUI as label engine → learned QC detector (on-the-go model training).** The
  hand-tuned `qc_*` rule is the bootstrap; the successor is a learned `P(error)`
  model whose training data is the *exhaust* of M4 review. Every correction a human
  makes is a label, so M4 logs one flat per-frame row — the QC signal vector
  (features), human verdict + error_type, the chain's anchor verdict, the frame's
  role, and whether the *rule* flagged it — into a per-frame label store M4 owns
  (the schema `calibration.py` sketched, now that the manual tool is shelved), live.
  Payoff: replace four hand-thresholds with one sliding probability knob (a clean
  precision/recall dial). This **dissolves the GUI-vs-calibrate ordering** — the GUI
  *is* the calibration instrument, the gold set is its byproduct, and the model is
  judged by §4's only metric: does it shrink the triage queue. Considerations, each
  a way the naive version quietly fails:
  - **Selection bias is the killer.** Labels collected only on *flagged* frames are
    censored: the model can learn to distrust the rule (cut false positives) but can
    *never* learn to catch what the rule misses (silent errors), because it sees no
    examples of the stable-but-wrong regime. M4 must therefore log a random sample of
    *un-flagged* "good" frames too (the `sampled` role above), even though there's
    nothing to correct on them. Non-negotiable — without it the model only ever
    shrinks the queue, never widens coverage.
  - **Anchor contamination poisons labels.** A frame wrong because the seed was wrong
    is not a propagation-signal failure; training on it teaches the model to predict
    anchor failures from features that can't see anchors. Log the anchor verdict;
    exclude (or separately model) bad-anchor chains.
  - **Feedback loop / covariate shift.** Each time you tighten the operating point you
    stop seeing the frames you now suppress, so the training distribution drifts under
    you (a filter trained on its own filtered output). Guard with a held-out,
    randomly-sampled eval set the model's decisions never touch — the only honest
    "is it improving" signal.
  - **Features bound the model.** It can only catch errors expressible in the signals;
    the stable-wrong case needs richer features. `pred_iou` — likely one of the strongest
    features and the one that targets the stable-wrong case — is **now populated** (§5 #3,
    June 2026), so it is available to log from the first M4 frame; calibrating its floor on
    this data is the remaining step (§7 *QC thresholds*).
  - **Keep modeling boring.** Logistic regression or a small gradient-boosted tree
    over the handful of signals; data volume is not the constraint, label *coverage*
    is. Split train/test **by chain or neuron** — adjacent frames share temporal
    signals, so a frame-level split leaks and inflates accuracy.
  - **Active learning is the real payoff.** Once a model exists, let its *uncertainty*
    (P≈0.5) choose what the human labels next, instead of relabeling whatever the rule
    already flagged — far fewer labels for a better detector. Conservative rule =
    turn-1 bootstrap; model uncertainty = turn-2 labeler.
- **[M3.5 auto · human→M4]** **Anchor selection:** mid-frame is the current heuristic; should a failed anchor
  re-pick (e.g. next node toward the chain center) automatically before queueing
  a human?
  *(Update June 2026: the gate that would drive a re-pick now exists in scoring-only
  form — `score_anchor` records the verdict but does not yet act. The automatic re-pick
  remains the open piece; it moves out of the `# [M4]` markers into the gate when the
  gate goes from observational to acting.)*
- **[M5]** **Chain merge conflicts:** when two chains of one neuron disagree on a z-slice,
  is the aggregate a plain union, or does overlap/voting matter?
- **[M3.5]** **Anchor prompt quality + a pre-propagation gate (raised, skeptical, unmeasured).**
  First-pass image-mode anchors — a single skeleton point + a few neighbour-cell
  *centroid* negatives, on the SCALE=8 frame — are often poor on thin neurites (a
  process is only a few px wide at 8×, below what SAM reliably segments). Cheap
  levers that need *no* model change: (a) run image mode on a full-res / 2× **crop**
  around the node instead of the 8× full frame, then downscale the box/mask to SCALE
  for the video seed (decouples anchor precision from propagation VRAM); (b)
  synthesize a **box** prompt from the CATMAID `radius` column (already in the node
  table) rather than a bare point — a box encodes extent, a point doesn't; (c)
  `multimask_output=True` + auto-select by node-containment / plausible-area /
  single-CC. Separately, an **anchor-quality gate** *before* propagation (score the
  anchor → auto-escalate prompts / re-pick the node toward chain centre → queue a
  human only on repeated failure) would make a bad anchor cost one frame's compute
  instead of a wasted ~300-frame propagation. This is the concrete form of the
  *Anchor selection* item above. Treat as unproven: measure the auto-anchor success
  rate first (the M3 batch is the instrument) and, per §4, invest only if it shrinks
  the queue enough to pay for the code.
  *(Update June 2026: lever (a) — the full-res crop — **landed as the default**, not as a
  gated escalation (see §6 M3.5 and §4). The gate itself (`score_anchor`) is wired and
  recording but still observational. Lever (c) `multimask_output` auto-select **landed,
  default-off** (`multimask_anchor`; `_select_anchor_mask` ranks the 3 candidates by
  node-containment → plausible-area → single-CC → decoder IoU). Verified near-free against
  the SAM2 source: the mask decoder computes all 3 candidate masks regardless of the flag
  and only slices the output (`sam2/modeling/sam/mask_decoder.py` `forward`), and the heavy
  image-encoder `set_image` runs once either way — so the "3× slower" worry does not hold;
  the only added cost is CPU-scoring 3 masks, and it touches only the one-frame anchor, not
  the ~300-frame video propagation. Lever (b) box-from-`radius` is **dead** — the CATMAID
  `radius` column is mostly placeholder values (decided June 2026).)*
- **[M4.5 model swap · napari-plugin eval→M4]** **EM-finetuned SAM / micro_sam — considered, deferred.** Domain-adapted SAM models
  segment EM neurites markedly better than vanilla SAM2 (the natural-image domain gap
  is real and well documented). Deferred anyway because: (1) it is *not* a localised
  swap — this pipeline's core is SAM2's **video** propagation, while micro_sam's
  finetuning targets the interactive **image** SAM, so it doesn't drop into
  `build_sam2_video_predictor`; at most it improves the anchor/image phase and forces
  a second model path. (2) It's still not 100%. (3) Per §4, accuracy isn't the
  bottleneck — supervised-per-chain already beats manual ~50×, so the refactor cost
  outweighs the gain right now. Decision: stay on vanilla SAM2; revisit only if
  measured failure rates make it pay. Orthogonal note: micro_sam's **napari plugin**
  (interactive EM prompting + annotation/correction, finetuned EM checkpoints) is a
  build-vs-adopt candidate for the **M4 GUI**, independent of any model swap — worth
  a look before writing a correction GUI from scratch.
- **[M3.5]** **Mask post-processing (new idea; cheap, deterministic, no model).** Saved masks
  are downscaled then nearest-neighbour upscaled, so they come out blocky / speckled
  / holey, whereas true neurite borders are smooth — so cleanup priors are safe to
  apply. A candidate `postprocess_mask` phase (§3a) between `propagate` and
  `save_masks`: largest-connected-component keep (generalise the one already in
  `box_from_mask`), morphological **open** (kill speckle) + **close** (bridge grid
  gaps), `binary_fill_holes`, and light boundary smoothing (morphological, or contour
  approximation / distance-field re-threshold). Open questions: apply at SCALE space
  (small kernels) or post-upscale (larger kernels); tune the kernel so thin neurites
  aren't erased (same failure mode as `qc_skeleton_dilation_px` set too tight);
  and decide ordering vs QC — clean *then* QC so QC scores the delivered mask, but
  watch that cleanup doesn't paper over a real propagation failure the QC signals are
  meant to catch.
  *(Update June 2026: **landed as a phase, default-off.** `pipeline.postprocess_mask`
  (model-free, `_sam`): morphological **open** (despeckle) → **close** (bridge NN-upscale
  grid gaps) → **largest-CC keep** (generalises `box_from_mask`'s pick) → `binary_fill_holes`.
  Decisions taken: (a) applied at SCALE/`_sam` space (small kernels — `postproc_open_px` /
  `postproc_close_px` default 1; a kernel bigger than the neurite half-width erodes thin
  processes, same failure mode as `qc_skeleton_dilation_px` too tight); (b) **ordered before
  QC** — folded into the save step so the saved PNGs are the cleaned masks and `run_qc`
  scores the delivered mask. Open caveat kept live: `postproc_keep_largest_cc` (default
  True) drops genuinely-split components in the *saved* mask and can mask a real propagation
  failure, so watch the flag distribution when enabling and consider False for chains where
  a process legitimately leaves/re-enters the plane.)*
- **[tier 1 DEFAULT · tier 2 LANDED default-off (June 2026) · tier 3 → later]** **Local
  high-res cropping (prior art: Bader Lab `sam2maskpropagator`).** Attacks the core accuracy
  problem (a neurite is ~3 px wide at scale 8). **Tier 1 (anchor-only crop) is now the default image phase**
  (June 2026): `run_chain` loads the full-res anchor frame, crops a `crop_size_tif`
  (default 1200 px) window around the node via `alignment.CropWindow`, runs image mode in
  `_crop` at `crop_scale` (default 2 → ~600 px input, near the old 8× cost but ~6× the
  linear resolution on the neurite), and maps the largest-CC box `_crop→_sam` for the
  video seed. The promised **single centralised transform** is `CropWindow`
  (`around_node` / `tif_to_crop` / `crop_to_sam` / `box_crop_to_sam` / `slice_tif`), which
  sidesteps the Bader x/y-swap trap (§4/§5) — the row/col swap lives only in `slice_tif`,
  verified by a marker round-trip test. **Measured (AVAL + the clean 5-neuron run):**
  sharpens the seed (cleaner anchors, tighter `box_sam`) but leaves the downstream
  `noskel` queue ~unchanged — as expected, since an anchor-only crop does not touch
  propagation resolution. Compute note: the default path adds one full-res `imread`
  (~240 MB at ~9k²) per chain's anchor, freed after; a windowed/memmap tiff read is a
  later optimisation. **Tier 2 — per-chain propagation crop — LANDED default-off (June 2026,
  supervisor-authorized accuracy-first; brought forward from M4.5 at the lab's request).** A
  new space `_pcrop`: `run_chain` crops ONE window sized to the chain's whole skeleton
  xy-extent (+ `chain_crop_pad_tif`, default 64) and runs the *entire* image phase **and**
  propagation inside it at `chain_crop_scale` (default 2), instead of the scale-8 full frame —
  the lever that actually moves downstream propagation resolution (tier 1 only sharpened the
  seed). Knobs on `PipelineConfig`: `chain_crop` (master switch, **default False** → the _sam
  full-frame path and the M1 baseline are unchanged), `chain_crop_pad_tif`, `chain_crop_scale`
  (a *target* — bumped coarser per chain so the input's longest edge stays ≤ `chain_crop_max_px`,
  default 1536, bounding VRAM for a chain that wanders far) and `chain_crop_min_tif` (default
  1024 — a **floor** on the window extent: a low-motion chain whose xy-bbox is tiny otherwise
  over-zooms and SAM2 loses inter-frame context; see the A/B below). Implementation reuses the
  single `CropWindow` home (new `around_box` builder + `sam_to_crop`; the only `[y,x]` swap
  stays in `slice_tif`). **Masks are stored in `_pcrop`** (the resolution win is kept on disk, for
  Blender), and the `CropWindow` is persisted to `state.json` (`ChainState.crop_window`) so
  QC, `review`, and the GUI rebuild the crop space: `qc.compute_metrics` maps skeleton nodes
  `_tif→_pcrop` via the window (the `scale==save_downscale` guard is skipped in crop mode, the
  containment radius rescaled by `scale/crop_scale`), and the napari GUI reconstructs the
  window for skeleton overlay + `--hires-em` (everything it shows — frames, masks, clicks —
  already shares the `_pcrop` grid, so a click is a `_pcrop` coord and re-predict/resume need
  no transform; this also incidentally gives tier-2 chains the **crop-space re-predict** the
  M4 GUI had deferred). Frame prep (`prepare_chain_crop_frames`) crops each frame the SAME
  crop-then-downscale way as the anchor, so seed and propagated frames share exact `_pcrop`
  pixels; it loses the cross-chain decode cache (each window is unique → one full-res imread
  per frame; windowed/memmap read is the documented follow-up). Window math + crop-aware QC
  are unit-tested (`tests/test_alignment.py`, +4 cases; 17/17). **A/B MEASURED (June 2026,
  real SAM2, RTX 3050 6GB, `ab_tier2.py` harness, 3 AIYL chains, large model, tier-2 on vs
  off):** tier-2 moved **2/3 chains `flagged`→`done`** — c12 (3 queued→0), c29 (3 queued→0,
  5 noskel→0) — at the crop's higher resolution (masks ~512px in `_pcrop` vs a ~3px-wide
  neurite speck at scale-8; ~17–30× the foreground pixels describing the same process), and
  the 3rd chain (c02, already clean) stayed clean — **no regression once the min-extent guard
  was in**. *The guard came directly from this A/B:* an **un-guarded first pass catastrophically
  failed c02** — a low-motion neurite (tiny xy-bbox) produced a 156×244 over-zoomed window where
  the anchor scored 0.52 and propagation collapsed to **empty masks on 29/39 frames** (→ 30
  queued). Adding `chain_crop_min_tif=1024` (floor the window, pad out for context) recovered it
  to anchor 0.90 / 0 empty / `done`. **So: tier-2 is a strong per-chain lever (eliminates the
  `noskel` queue on chains that have xy-motion) but NOT safe to enable blindly — over-zoom on
  low-motion chains is a real failure mode; the min-extent floor mitigates it, and the
  `image_score`/anchor gate should guard a per-chain fall-back to `_sam` (next step).** Cost:
  tier-2 ran ~2× the wall-time of baseline here, dominated by the per-frame full-res `imread`
  in frame-prep (propagation itself is *cheaper* than the full frame) → the windowed/memmap
  read is the priority optimisation. Verified non-degenerate + visually (overlay grids in
  `ab_figs/`). **Open next:** (a) image_score/anchor-gated auto fall-back to `_sam` when a crop
  anchor is poor; (b) tune `chain_crop_min_tif` (1024 slightly relaxed c12's tight win —
  0 queued either way); (c) the windowed/memmap frame read; (d) wider A/B across neurons +
  M4-label confirmation that the remaining single-signal `noskel` is benign.
  *(Update June 2026 — items (a) and (c) landed; ab_fallback.py.)* **(a) anchor-gated fall-back to
  `_sam`** is in (`chain_crop_fallback`, default on): when a chain's `_pcrop` anchor is poor it
  re-runs the whole chain in the plain `_sam` path instead of propagating a collapsed crop.
  **Key correction from the A/B:** the geometry gate alone does NOT catch the over-zoom — the
  over-zoomed anchor *passes* it (clean blob, contains the node); the collapse is a propagation
  effect, invisible at the anchor frame. The discriminating signal is SAM2's anchor `image_score`
  (over-zoom **0.516** vs healthy **0.848 / 0.879**), so the fall-back fires on
  `chain_crop_min_image_score` (default **0.7**, first-pass — tune in the wider A/B). Verified: the
  forced-over-zoom chain falls back and recovers the clean baseline (status `done`, 0 queued, _sam
  dims) at no extra wall-time; a good-crop chain (0.879) stays tier-2 and does not regress
  (`fell_back_to_sam` recorded on ChainState for the P(error) features). **(c) windowed/memmap
  frame read** is in (`_read_tif_window`): a `tifffile.memmap` row-window slice (the EM tifs are
  uncompressed single-strip 8-bit grayscale → memmappable), with a full-`imread` fallback for any
  tif that isn't. Measured **bit-identical** to `cv2.imread(tif)[sl]` and **~48× faster** per frame
  (566→12 ms), so the ~2× tier-2 wall-time penalty above is essentially gone. Still open: (b) tune
  `chain_crop_min_tif`; (d) wider A/B across neurons + the `image_score` floor's true value.
  **Tier 3 → later:**
  (3) **per-frame tracked** crop following skeleton xy(z) — max resolution, but a shifting origin
  per frame (per-frame remap; may help by centring the object, may confuse tracking —
  speculative); still unbuilt.
- **[M3.5 auto seed · human anchor→M4]** **Video seed: box vs mask (confidence-gated), incl. human-painted anchors.** The box
  doesn't *avoid* needing an accurate anchor mask — it *delegates* making one to SAM2:
  a box seed has SAM2's decoder produce the anchor mask and stores *that* in the memory
  bank, and that box→mask step is exactly the single-image guess we've shown is
  unreliable on thin EM neurites at 8×. A mask seed (`add_new_mask`) bakes a curated
  boundary into memory instead — strictly more informative *when it's right*, but memory
  propagates faithfully, so a slightly-wrong mask propagates its error whereas a box
  lets SAM2 re-derive something plausible. Hence a **confidence gate**: seed with the
  mask when the anchor is trustworthy (QC-pass / high `image_score` / human-touched),
  else fall back to the box. Per §4 (accuracy + HITL over automation-%), the mask is the
  *target* seed; the box is the transitional default while anchors are still auto-and-
  rough. Nearly free to try — `image_predict` already computes the mask and the current
  pipeline discards it for its bbox, so just add a mask-seed path in `propagate` + the
  gate. A **human-painted anchor** is the maximally-verified case: for tiny / single-node
  / E-or-U chains where prompting fights you, the human paints the anchor and SAM2
  propagates the rest — automating the ~300-frame step while conceding the one hard
  frame. Per-chain routing: trivially small chains, or chains whose anchor QC fails after
  auto-retries, go to human-anchor rather than burning compute. Folds into the same M4
  mask-edit surface, and directly serves the supervisor's accuracy + HITL mandate (§4).
  *(Decision June 2026: **co-build this with the M4 GUI, do not build it pre-GUI.** Two
  reasons: (1) its highest-value mode — the human-painted anchor → mask seed — *is* an M4
  feature (the mask-edit surface), and (2) certifying "when does the mask seed beat the box
  seed" needs ground-truth labels, which only the GUI produces (§6 ruler; the M3.5 proxy
  ruler can't see silent errors). The mechanism is already in place — `PropagationSession.add_mask`
  exists and is AVAL-validated — so the auto/confidence-gated path is a thin add alongside
  the GUI's human-anchor path, not a separate build. This is the counterpart to the multimask
  decision, which we landed pre-GUI precisely because it is near-free and self-contained.)*
  *(Update June 2026 — **seed ablation landed + measured** (`ab_seed.py`; flexible
  `seed_box`/`seed_points`/`seed_negatives`/`seed_mask` + `box_margin_frac`). **API fact:** SAM2
  makes MASK and POINTS/BOX mutually exclusive per frame (`add_new_mask` pops `point_inputs` and
  vice-versa), so "mask + points on the anchor" is NOT a real config — the valid space is
  mask-only OR any subset of {box, pos, neg}. **Result (3 AIYL chains, 88 frames, anchor held at
  scale-8 `_sam` so seed type is isolated), ranked by queue:** `box_pos` (the current default) = 6
  queued, tied with the box+neg / boxfrac variants; `box_only`/`mask_only` = 8; `pos_only` = 9.
  **Takeaways:** (1) **box+positive is the best AUTO seed — keeping the box was correct**; point-only
  *regresses*. (2) **`mask_only` does NOT beat the box at scale-8** (8 vs 6) — confirms the mask seed
  only wins on a *high-quality* anchor (curated/human-painted or tier-2 `_pcrop`), so scrapping the
  box in the *GUI* (human paints a good mask) was right AND keeping it for scale-8 AUTO was right;
  the two decisions are consistent, not contradictory. (3) **Negatives are chain-dependent**, not a
  blanket win: c12 queue 4→1 with negatives, but c29 2→5 — helps concave/cluttered, hurts clean,
  net wash. Keep `seed_negatives` a targeted lever, default-off. (4) **`box_margin_frac` (underfill
  fix) — VALIDATED** (`ab_underfill.py`, scan 23 chains -> A/B the top-3 high-noskel suspects).
  **RIML c25 was a genuine underfill**: fixed-10px box -> noskel 9/21, queue 4, *flagged*;
  `box_margin_frac=0.5` -> **noskel 0, queue 0, *done*** (the size-relative pad enclosed the whole
  cell the fixed box clipped). Your bounding-box instinct was right — under-filled anchors are a
  real failure mode and the frac margin fixes them. BUT it's TARGETED, not universal: of 3
  high-noskel suspects only RIML c25 was true underfill; AIYL c12 (noskel identical across all
  seeds) was tracking drift and AVBR c12 (img_score 0.27) a poor anchor — frac was inert on both,
  and `mask_only` was WORSE on all three (re-confirming the mask seed needs a good anchor). So keep
  `box_margin_frac` default-OFF and make it a **targeted retry lever**: a chain that flags with high
  noskel + a contained anchor is the signal to re-run it with the frac margin (the same
  retry-on-failure pattern as the item-b tier-2 fallback). Caveat: small sample, weak deltas
  (6 vs 8-9) — directional. Default seed unchanged (`box_pos` won); the knobs are additive.)*
  *(Update June 2026 — **wider tier-2 A/B** (`ab_tier2_wide.py`, 15 chains × AIYL/RMDR/AVBR, tier-2
  with the item-b fallback on): improved 3, **regressed 0**, unchanged 12, fallback fired 6/15, net
  queue −10. Tier-2-with-fallback only helps or stays neutral (via fallback) across 3 neurons → safe
  to enable on flagged chains; AVBR c12 (worst, queue 9) fell back to `_sam` and needs the GUI.)*
- **[M3.5 auto · manual→M4]** **Negative points in video seeding.** `add_new_points_or_box` takes labelled points
  *and* a box on the prompt frame, so adding negatives to the video seed is trivial.
  Most useful for concave shapes (E/U neurons) where a box bounds a concavity that
  belongs to a neighbour; auto-negatives can come from neighbour skeleton nodes (same
  source as image mode). Same mechanism the M4 GUI uses to correct a degrading frame.
  Open: whether neighbour-node negatives actually land in the concavities — measure.
  *(Update June 2026: **landed, default-off.** `propagate(..., seed_negatives=False)` /
  config `seed_negatives`. On = forward the same-z neighbour negatives `build_prompts`
  already computes in `_sam` to the video seed (the seed-time analogue of the image-mode
  negatives); off = positives-only = the M1 seed. The "do negatives land in the concavity"
  question is now an A/B switch — still unmeasured. Risk to watch: the k-nearest negatives
  can include the same neuron's other chains / nearby branches, so on concave E/U chains
  confirm they suppress the *neighbour*, not legitimate foreground.)*
- **[M4.5 session — June 2026 considerations]** Open ideas raised alongside the tier-2 work, recorded
  here so they aren't lost:
  - **GUI manual-paint resolution is bounded by the propagation space.** The paintable MASK
    (Labels layer) lives at whatever resolution the chain was propagated/saved in: scale-8 for
    M1/tier-1 `_sam` chains, `_pcrop` (crop_scale≈2) for tier-2. `hires_em=True` sharpens only the
    EM *background*, not the mask — so on a scale-8 chain a hand-painted stroke is an 8×-coarse grid
    that becomes 8×8 blocky blocks when upscaled to full-res output. This is inherent, not a bug:
    `add_new_mask` must match the frame resolution SAM2 is propagating, so you cannot paint finer
    than the frame space. **The fix is to paint in a higher-res space = tier-2** — its `_pcrop` view
    *is* the crisp mask-edit surface the lab wanted. So crisp manual paint and open step (e)
    (re-propagate a corrected `_sam` chain as tier-2) are the same lever. Keep manual paint; route
    chains that need hand-correction through tier-2.
  - **Mask-vs-box video seed + %-of-mask-width margin** (extends the §7 box-vs-mask bullet above).
    The box is still the headless AUTO seed; the GUI already seeds mask-only. Now that the GUI can
    produce ground-truth labels, A/B mask-seed vs box-seed under the same confidence gate as step
    (b). The fixed `box_margin=10` px is the suspect in the "box from an underfilled mask doesn't
    contain the whole cell" failure: replace/augment with a margin = X% of the mask's width so the
    box pad scales with cell size. If box-with-%-margin recovers the underfill cases, the box stays
    viable as the low-confidence fallback rather than being scrapped outright.
  - **Stricter flag (QC gate) params — shelved until the predictor exists.** Idea: open the flag
    thresholds very wide initially (catch every possible failure), then tighten as collected
    review-label data trains a predictor. Deferred: only worth tuning once the predictor that
    consumes the labels is actually being built, else we're hand-tuning thresholds twice.
  - **Split the GUI into a marking pass and an intervention pass.** Current single window has too
    many controls and lets the user scroll *all* frames even when only the root/center/flagged node
    should be reviewed — which confuses the seeding/resume logic. Proposed flow: (1) *marking mode* —
    load a chain's whole frame stack, reviewer marks good frames OK; (2) *intervention mode* —
    entered on hitting a bad frame, shows only the flagged/picked frame for correction. Constrains
    review to the frames the pipeline actually conditions on. For consideration; not yet scoped.
- **[M3 — landed]** **Runtime telemetry — landed.** `run_chain`'s `_step` now wraps each phase in
  `perf_counter`, accumulating per-phase seconds into `state.phase_seconds`; the batch
  driver brackets each chain with `diagnostics.reset_peak_vram()` / `peak_vram_gb()`
  (peak `torch.cuda.max_memory_allocated`) and appends one row per chain to
  `output/_timing.csv` — `neuron, chain_idx, n_frames, peak_vram_gb, t_<phase>…, t_total`.
  Fixed phase-label schema so appended rows never misalign; the write is wrapped so a
  telemetry hiccup can't kill a chain; review functions stay untimed (human-paced).
  Placement note vs. the original plan: the timer lives in `pipeline.run_chain` on
  stdlib `perf_counter` (keeps `pipeline.py` torch-free, same reasoning as the
  `on_video_phase` callback), `diagnostics` owns only the VRAM probes, and the batch
  driver owns the CSV. This is the prerequisite the speed items below were waiting on:
  one overnight run now yields time-vs-`n_frames` per phase (expect `propagate` to
  dominate, ~linear in frames now that prep is cached) and per-chain VRAM high-water
  marks (the headroom number for the GPU / multi-GPU questions). *(No longer open —
  just read the numbers off the next batch before any speed/hardware work.)*
- **[infra · parallel-review→M4]** **Performance scaling (GPU / multi-GPU chain sharding).** Propagation is GPU-compute
  + VRAM bound and sequential *within* a chain (the memory mechanism), so a faster GPU
  with more VRAM helps directly (more VRAM → less CPU offload). The big lever: chains
  are independent, so multi-GPU **chain-sharding** is the clean scale-out — each worker
  atomically claims a `pending` manifest row, runs it, marks it done; the resume design
  is already most of a work queue. Single-GPU multiprocessing mostly contends for one
  card — skip. Caveat (per §4 + supervisor): with a human reviewing flagged frames the
  human is the throughput limiter, so GPU speed shortens the *unattended* pass, not
  wall-clock to a finished dataset. Measure (telemetry) before buying hardware or
  building a multi-GPU harness.
  *Parallel review + background compute:* run the batch and the review GUI
  concurrently — background works `pending` rows (producer: segment + flag), the human
  works `flagged` rows (consumer), manifest = the shared queue. Wall-clock becomes
  max(GPU, human) instead of the sum, and the human is never blocked: by the time a
  reviewer clears the current flagged batch, more chains are done and freshly flagged.
  Tightens the test/refine loop too. Needs three things, all filesystem-only (no
  server/db): (a) **concurrency-safe manifest** — partition ownership (background owns
  *execution* status pending→running→done/flagged/failed; GUI owns a separate *review*
  status column) + a file lock (`filelock`/`portalocker`) around writes; (b) the GUI
  **polls/watches** the queue so chains flagged mid-session appear; (c) **GPU
  arbitration** for interactive re-runs (a human correction → `add_new_mask` →
  re-propagate competes for the card) — interleave on one GPU (corrections are
  intermittent), or, under multi-GPU sharding, dedicate one GPU to interactive and the
  rest to batch. So parallel-GUI and multi-GPU are the same architecture from two angles.
- **[M4.5-ish] Marking/intervention GUI split (review-testing feedback, June 2026).** The
  single dense panel is confusing and lets the reviewer scroll to any frame and act on it,
  which muddies what the system thinks is being reviewed. Proposed two-mode flow: a
  **marking** mode that loads a chain and lets the human sweep frames ok/bad (label-only,
  no edits), and a separate **intervention** mode entered on a bad frame that shows *only*
  the flagged/selected frame(s) and exposes the correction tools (points / paint / resume).
  Cleaner than the current all-in-one dock and removes accidental-edit-while-scrubbing.
  Not built — the current single-panel GUI works; do this before or with M4.5 at the lab's
  discretion. Pairs naturally with the §6-row-4 deferred work.
- **[before M4.5] Strict-by-default flagging (review-testing feedback, June 2026).** Operating
  posture, not a mechanism change: set the `qc_*` thresholds to flag **aggressively** (high
  recall — catch every plausible error, tolerate false alarms) for the first labeled
  campaign, then loosen once the M4.5 learned `P(error)` detector has labels to set the
  operating point against ground truth (the *GUI as label engine* item). Concretely: tighten
  `qc_area_ratio_bounds` (e.g. `(0.7, 1.5)`), raise `qc_temporal_iou_min` and `qc_pred_iou_min`,
  and set `qc_triage_min_signals = 1` (queue every flag, not just intervene). NB the §5#7
  mixed-threshold discipline: **clear/re-score the manifest** after changing thresholds, or
  early and late chains silently mix two configs. Until the learned detector exists, the
  rule's job is recall, not precision — the human is the precision filter, and every
  decision is a label.

---

*Update this doc as decisions land — it's the shared big-picture reference.*