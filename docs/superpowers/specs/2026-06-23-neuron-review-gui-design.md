# Neuron-level review GUI (interactive whole-neuron tracing)

Date: 2026-06-23
Status: draft for review

## Motivation

The current pipeline decomposes each neuron's CATMAID skeleton into linear chains and
tracks each independently with SAM2. It works, but the human reviews one chain at a
time, and the annotator (the supervisor) thinks in terms of *neurons*: click a neuron,
see it traced, fix it. She does not think in chains, and some chains are artificial
splits (a "clarity node" on a still-connected neurite), which fragments one neuron's
identity across several review units.

The error mix is shape-dependent and bidirectional: irregular/concave/hole-y neurites
tend to be under-filled, small/ill-defined ones over-filled. No single automatic lever
fixes both, and post-processing cannot grow an under-filled mask. So a human in the loop
is genuinely the right tool, and the unit she works in should be the neuron.

This is a second review paradigm, not a replacement. The per-chain GUI (`gui.py`) stays
exactly as is; this is a new, parallel driver.

## Decisions already settled (brainstorming)

- **Branches stay separate objects; the neuron is a presentation + union layer.** We do
  NOT track a whole neuron as one multi-blob SAM2 object (verified failure risk: SAM2's
  per-object memory drops/merges branches; see the sam2-multiobject memory note). Each
  branch is its own SAM2 object (today's per-chain tracking, unchanged). The multi-blob
  neuron result on a branch-point slice is produced by *union*, not by tracking.
- **Identity grouping is free.** A neuron = the chains sharing a CATMAID `cell_name`.
  Over-split "clarity node" chains dissolve automatically: same `cell_name` means same
  identity means unioned.
- **Separate script.** A new `gui_neuron.py`, reusing the library and the shared, proven
  pieces of `gui.py` (ReviewContext, frame loaders, the box/prompt helpers,
  PropagationSession wiring). `gui.py` is not modified.

## Architecture

### Layers (napari)

- **EM** image over the neuron's union z-range, on the canonical `_sam` grid.
- **neuron** Labels layer: ONE layer holding an integer per branch (branch 1 = 1,
  branch 2 = 2, ...). Each integer renders a distinct colour; every nonzero pixel is
  "the neuron". This is the key simplification: napari Labels layers are multi-label by
  design, so branches do not need separate layers, and `selected_label` is the
  active-branch selector (with `show_selected_label` to dim the rest). The per-branch
  integers are an editing convenience; the saved identity is the neuron.
- **prompts** Points + **box** Shapes: single layers, scoped to the active branch (the
  selected label). They are ephemeral per-correction, so there is never a need for
  per-branch prompt layers.

### Building the neuron view

Each branch's saved masks are remapped onto the common `_sam` grid with
`pipeline.chain_masks_in_sam` (already handles both legacy `_sam` and tier-2 `_pcrop`
branches, with their offsets). Paste each branch into the neuron Labels volume under its
own integer. Union for display is automatic (nonzero); cross-neuron overlap arbitration
(the `non_overlap` argmax) is applied only between different neurons at save, which is
the ERL win and is independent of editing.

### Editing model (the active branch)

A correction (points, box, paint, recrop) acts on the selected branch only, through the
same proven mechanisms the per-chain GUI already uses, run in that branch's native space
(`_sam` or its `_pcrop`); the result is written back to that branch's integer in the
neuron layer via the same `chain_masks_in_sam` remap. Each branch is backed by its own
`PropagationSession` keyed by `obj_id`, built lazily when the branch becomes active and
closed when switching away (one live session at a time, the safe default for VRAM).

### Disposition

Approve / reject / correct at the **neuron** level. The review-queue ledger gains a
neuron-grained view (or aggregates its chains' statuses); a neuron is done when all its
branches are dispositioned.

## Reuse vs new

- Reused from `gui.py` (imported, not duplicated): `ReviewContext`, frame-stack loaders,
  the pure box helpers, prompt/box layer builders, the correction actions
  (`rerun_image_phase`, `resume_propagation`, recrop), label logging.
- Reused from the library: `chain_masks_in_sam`, `PropagationSession`, `run_chain`,
  `review.load_chain`, `review_queue`, `labels`.
- New in `gui_neuron.py`: neuron enumeration (chains by `cell_name`), the multi-label
  neuron Labels layer + active-branch selection, per-branch session lifecycle, the
  neuron-level disposition, and building/refreshing the neuron view from branch masks.

## Open decisions for review

1. **Editing surface.** Two viable v1 shapes:
   - *Unified canvas (recommended):* edit any branch in place on the neuron view; when a
     branch is selected, show its native-space frames for the correction and write the
     result back to the neuron layer. Most faithful to "see and fix the neuron"; the cost
     is the display(`_sam`)-vs-edit(native) space bookkeeping.
   - *Overview plus drill-in (simpler):* the neuron view is a read-mostly `_sam` overview;
     selecting a branch opens it in the existing per-chain editor (native space), exactly
     today's flow, then returns to the overview. Lower risk, less seamless. Easy to ship
     first and grow into the unified canvas.
2. **Cross-neuron overlap arbitration in v1, or later?** It is the ERL win and is
   independent of the editing UX, so it can land as a save-time step in v1 or be deferred.
3. **Mixed-space neurons.** A neuron often has both `_sam` and tier-2 `_pcrop` branches.
   The overview is unified on `_sam` (downsampling tier-2 for display). The question is
   only whether a *correction* on a tier-2 branch happens at `_pcrop` resolution (more
   code) or is allowed to drop to `_sam` for v1 (simpler, loses tier-2 sharpness while
   editing). Recommendation: keep tier-2 corrections in `_pcrop` by reusing the per-chain
   editor's native-space path (which already does this), which is another reason the
   overview-plus-drill-in shape is attractive for v1.

## Non-goals

- No change to `gui.py` or to per-chain batch processing.
- No multi-blob single-object tracking (rejected).
- No new segmentation model; this is a review/correction tool over existing SAM2 output.

## Testing

- Pure/logic pieces (neuron enumeration from `cell_name`, building the multi-label
  volume from per-branch `chain_masks_in_sam` outputs, active-branch read/write) are
  torch-free and unit-tested, like the existing crop/queue tests.
- The napari wiring stays untested in CI (no napari there), matching `gui.py`; verified
  by a manual launch.
