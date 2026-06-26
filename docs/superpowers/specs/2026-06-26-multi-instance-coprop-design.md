# Multi-instance co-propagation: testing the neighbor-competition hypothesis

Date: 2026-06-26
Status: design, pending implementation

## Background and hypothesis

The PI's hypothesis: when we segment a neuron chain, also segmenting and propagating its
neighbors might improve the mask of the chain we actually want. The intuition is that the
neighbors hold the target in check, so it stops bleeding into territory that is not its own.

This document records the design for a focused, visual test of that hypothesis on the
sensory-ablated target worm, wired into the per-chain review GUI (`gui.py`) behind a new
action that leaves every existing flow untouched.

## The mechanism (what the test must respect)

The codebase today propagates one object per chain: `run_chain` drives a single
`state.obj_id`, and `PropagationSession.__init__` calls `reset_state` per object, so each
chain runs in its own isolated `inference_state`. Neighbors enter only as negative point
prompts at the anchor (`build_prompts` already gathers same-z neighbor nodes).

Reading the installed SAM2 source settles how neighbors could ever change the target.
SAM2's multi-object tracking runs the memory attention independently per object: object A's
mask never informs object B's propagation. The only coupling is
`_apply_non_overlapping_constraints` (`sam2/modeling/sam2_base.py`), a per-pixel argmax over
object scores that keeps the highest-scoring object at each location and suppresses the rest
to `sigmoid(-10)`.

So the only way segmenting neighbors can change the target's mask is through that argmax.
With the constraint off, the target mask is bit-identical whether or not neighbors are
tracked. There are two flavors of the constraint:

| Flag | Where it acts | Effect |
|---|---|---|
| `non_overlap_masks` | output masks only | Post-hoc cleanup, identical to a labelmap argmax we already need at compositing. Does not change propagation. |
| `non_overlap_masks_for_mem_enc` | masks fed back into memory | Changes what each object memorizes, so it alters the propagation trajectory across z. The only variant that can improve tracking, not just clean the final frame. |

Both are plain attributes on the video predictor, settable at runtime and restored after use.

Why the hypothesis is plausible here specifically: the Stage 0.2 benchmark found the pipeline
merge and bleed dominated (precision near 2.5%). The target mask claims pixels that belong to
neighbors. If those neighbors are tracked too, a neighbor that scores higher on a contested
pixel takes it back, so the argmax carves the bleed out. That is a concrete, mechanism-grounded
path to higher precision, not a vague appeal to richer context (there is no richer context).

## Goal and non-goals

Goal: prove or disprove the hypothesis with a clean A/B on the target worm, documented
visually, one chain at a time, inside `gui.py`.

Non-goals (deferred, not part of this work):
- No ground-truth scoring. The chosen evaluation is visual on the target worm. If the result
  is promising, the GT-scored version reuses the library components below unchanged.
- No batch driver and no new GUI window.
- No change to `run_chain`, `PropagationSession`, or any existing GUI behavior. The new action
  is additive and gated.

## The A/B design

The only way neighbors can act is the non-overlap argmax, so the cleanest control propagates
the target in the same multi-object session both ways and flips one switch:

- Baseline: target plus neighbors seeded, non-overlap OFF. With the constraint off there is no
  coupling, so the target mask equals the target propagated alone. This is a free, perfectly
  matched control built from the same seeds.
- Treatment: identical seeds, non-overlap ON (default `non_overlap_masks_for_mem_enc=True`,
  with the output-only variant available for comparison).

The only variable between the two runs is the constraint, so any difference in the target mask
is unambiguously the neighbor-competition effect.

## Components

### 1. `MultiObjectPropagationSession` (library, new)

Lives alongside `PropagationSession` in `pipeline/propagate.py`. The existing single-object
class is not modified, so the production path stays bit-identical.

- Holds one `inference_state`.
- Seeds N objects from a mapping `{obj_id: (prompts, box, anchor_frame_idx)}`.
- Sets `video_predictor.non_overlap_masks` and `non_overlap_masks_for_mem_enc` on entry and
  restores both on close, the same set-and-restore discipline the IoU hook already follows, so
  a shared predictor is never left mutated.
- Propagates bidirectionally and returns `{obj_id: {frame_idx: mask}}`.
- Reuses the existing IoU hook and confidence collection per object where it applies.

### 2. `neighbor_chains(...)` (library, new pure function)

In `pipeline/predict.py` near `build_prompts`. Pure and torch-free, so it is unit-testable
under the repo's CPU-only test rule.

- Inputs: the target chain, `annotate_df`, `chains`, the propagation window and z-range, scale.
- Returns the k nearest other chains (default 3) that have at least one node falling inside the
  propagation frame on a shared slice. Chains with no in-window node cannot contend and are
  dropped.
- Maps neighbor node coordinates into the propagation space with the same transform the target
  used (scale for the legacy `_sam` path, the crop window for tier-2 `_pcrop`).

Each returned neighbor is seeded through the same anchor path the target uses (`build_prompts`
plus a box from `image_predict` or `anchor_crop_predict`), so neighbors get real masks that can
genuinely contend, rather than a degenerate point seed that would under-contend and muddy the
test.

### 3. New gated action in `gui.py`

A button plus key on the existing per-chain GUI, built on `ReviewContext.video_predictor` and
the open chain's `state.frames_dir`.

- Suggests the k nearest neighbor chains (k selectable in the dock for testing).
- Lets the user manually add or remove neighbor chains before running (pick from the in-window
  chain list, or click near a neighbor's node on the canvas to add it). The auto-suggestion is
  a starting point, not a fixed set.
- Runs the session twice over the same seeds: non-overlap OFF, then ON.
- Adds new napari layers without touching the existing `mask` layer or any existing flow:
  `mask (alone)`, `mask (w/ neighbors)`, `neighbors` (distinct colors), and a `diff` layer
  highlighting the target pixels gained and lost between the two runs.
- The user toggles layer visibility for the before/after comparison and documents it with
  napari screenshots.

When the action is not invoked, `gui.py` behaves exactly as it does today.

### 4. Tests

Torch-free unit tests for `neighbor_chains`: selection by distance, the in-window filter, the
k cap, and the coordinate mapping for both the `_sam` and `_pcrop` spaces. The session is
torch-bound, so it is exercised manually through the GUI rather than in the CPU suite, matching
the repo's existing test split.

## What proves or disproves the hypothesis

- If the `diff` shows the target shedding bleed into neighbor territory when the constraint is
  on, the PI is right, with a mechanism to explain it.
- If the target mask is essentially unchanged, or loses real area to an over-eager neighbor, the
  hypothesis does not hold for this pipeline, and the overlays show why.

## Risks and caveats

- The mem-enc variant changes the trajectory, so a neighbor with a poor mask could steal real
  target pixels. The OFF-vs-ON A/B and the `diff` layer make that failure visible rather than
  hidden.
- Neighbors must actually fall inside the propagation window to contend. The in-window filter
  enforces this, and a chain whose window is tight may surface few or no neighbors, which is
  itself an informative result.
- Coordinate space is the usual hazard: neighbor nodes must be mapped into the same space as the
  target's frames. The mapping reuses the target's own transform to avoid drift.

## Out of scope

GT scoring, batch runs, a dedicated viewer, and any change to the single-object production path.
