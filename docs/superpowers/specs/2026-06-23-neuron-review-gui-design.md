# Neuron-level review GUI (interactive whole-neuron tracing)

Date: 2026-06-23
Status: approved, ready for an implementation plan

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

### The canvas: a per-neuron crop (`_ncrop`)

The unified canvas is a single per-neuron crop window, `_ncrop`, sized to the whole
neuron and shared by every branch. It is the tier-2 chain-crop machinery lifted to
neuron scope: take the skeleton bbox of ALL the neuron's chains (generalize
`_chain_skeleton_box_tif` to the union of their node sets), pad it, and pick an adaptive
`crop_scale` that keeps the input edge under `chain_crop_max_px`. Every branch, `_sam`
or tier-2, remaps into this one grid, so the mixed-space problem is eliminated rather
than managed: there is one space for the whole neuron.

Resolution is the inherent trade. A view's pixel resolution is `tif / downscale`:
`_sam` is `tif/8`, a tier-2 chain crop is `tif/crop_scale` (default `tif/2`, 4x `_sam`).
`_ncrop`'s `crop_scale` adapts to the neuron's extent, so a compact neuron gets a sharp
canvas (better than `_sam`) and a sprawling one approaches `_sam`. `_ncrop` is never
worse than the full-frame `_sam` fallback, but for a large neuron it is coarser than a
single branch's own tight tier-2 crop. That gap is why the per-chain GUI stays: it is
the place to fix one thin branch at full tier-2 sharpness.

Cost to budget: opening a neuron prepares full-res tif windows over the union z-range
(the per-frame windowed read recrop already uses). It is a one-time cost per neuron-open,
shared across all branches, but the view does not pop open instantly.

### Layers (napari)

- **EM** image over the neuron's union z-range, in `_ncrop`.
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

Each branch's saved masks are remapped into `_ncrop` and pasted into the neuron Labels
volume under its own integer. The remap is a small generalization of
`pipeline.chain_masks_in_sam` (which already goes native `_sam`/`_pcrop` -> `_sam`) to
target an arbitrary window instead of `_sam`. Union for display is automatic (nonzero).
Cross-neuron overlap arbitration is deferred (see Decided scope).

### Editing model (the active branch)

A correction (points, box, paint, recrop) acts on the selected branch only and runs in
`_ncrop` over the shared neuron frames, through the same proven mechanisms the per-chain
GUI uses (`rerun_image_phase` / `resume_propagation`). Because all branches share
`_ncrop`, the neuron frames are prepared once and reused; the branches can live as
separate `obj_id`s in a single SAM2 `inference_state` over those frames (the correct use
of multi-instance: shared image encoder and frames, not shared context). A correction
writes its result back to that branch's integer in the neuron layer, and the branch's
saved masks become `_ncrop` (its `state.crop_window` is the neuron window). Display is
always "remap each branch's current masks into `_ncrop`", so a branch not yet edited
still shows from its native-space masks; only the edited branch is re-run.

### Disposition

Approve / reject / correct at the **neuron** level. The review-queue ledger gains a
neuron-grained view (or aggregates its chains' statuses); a neuron is done when all its
branches are dispositioned.

## Reuse vs new

- Reused from `gui.py` (imported, not duplicated): `ReviewContext`, frame-stack loaders,
  the pure box helpers, prompt/box layer builders, the correction actions
  (`rerun_image_phase`, `resume_propagation`, recrop), label logging.
- Reused from the library: `PropagationSession`, `prepare_chain_crop_frames` (pointed at
  the neuron window), `review.load_chain`, `review_queue`, `labels`.
- New in the library (`pipeline/crop.py`, pure + tested): a neuron crop window builder
  (skeleton bbox over all the neuron's chains, adaptive `crop_scale`), and a generalized
  remap that targets an arbitrary window instead of `_sam` (factor it out of
  `chain_masks_in_sam`, which becomes the `target = _sam` case).
- New in `gui_neuron.py`: neuron enumeration (chains by `cell_name`), the multi-label
  neuron Labels layer + active-branch selection, the shared `_ncrop` frames + one
  `inference_state` with branches as `obj_id`s, neuron-level disposition, and
  building/refreshing the neuron view from branch masks.

## Decided scope (v1)

1. **Unified canvas.** Edit branches in place on the neuron view; no drill-in to the
   per-chain editor.
2. **Cross-neuron overlap arbitration is deferred.** Within a neuron, branches union with
   no conflict, so v1 needs no arbitration. The cross-neuron `non_overlap` argmax (the ERL
   win) is a later save-time step, independent of this UX.
3. **Mixed-space is eliminated by `_ncrop`,** not managed: every branch remaps into the
   one neuron crop, and corrections run in `_ncrop`. The resolution trade (a large neuron
   gets a coarser canvas than a branch's own tier-2) is accepted; the per-chain GUI
   remains for full-sharpness single-branch work.

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
