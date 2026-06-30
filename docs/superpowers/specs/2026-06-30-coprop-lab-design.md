# Co-propagation lab: a standalone test for the neighbor-competition hypothesis

Date: 2026-06-30
Status: design, pending implementation

## Purpose

Test, visually and disposably, whether co-segmenting a chain's neighbors changes the target
chain's mask for the better. This is a throwaway experiment, not a pipeline feature. It lives in
one standalone file (`coprop_lab.py`), saves nothing, scores nothing, and leaves the pipeline
library and the production GUI byte-identical. When the question is answered, the file can be
deleted.

This supersedes the reverted auto-node attempt (branch `feat/multi-instance-coprop`,
findings in `docs/superpowers/2026-06-29-multi-instance-coprop-findings.md`). The two
failures of that attempt drive this design: it selected neighbors automatically and got the
wrong cells, and it never seeded the target from a correct mask, so it measured auto-seed
quality rather than the neighbor effect.

## The two questions, and why they are two flags

The only way one object can change another in SAM2 is the per-pixel argmax in
`_apply_non_overlapping_constraints` (`sam2/modeling/sam2_base.py:891`): at each pixel keep the
highest-scoring object and suppress the rest to a score of at most -10. Memory attention is
otherwise independent per object. That argmax has two flavors, and they are exactly the two
questions worth asking.

**Test 1, segmentation cleanup.** Does competition clean up the target's mask? This is the
output-only constraint (`non_overlap_masks`, applied at `sam2_video_predictor.py:401` in the
propagate yield path, after memory encoding). It carves the target's bleed wherever a neighbor
scores higher, but does not change the propagation trajectory. It pairs naturally with the
**current auto-seed**, because the question is whether the existing output gets cleaner.

**Test 2, propagation.** Starting from a correct seed, does co-propagation track better? This is
the memory constraint (`non_overlap_masks_for_mem_enc`, applied at `sam2_base.py:696`, before
memory encoding). The argmax-carved mask is fed back into each object's memory, so it changes
what every later frame conditions on, so it changes the trajectory across z. This is the only
variant that can make tracking genuinely better or worse. It needs a **correct seed** to be a
fair test: a bad seed propagates to garbage either way and hides the neighbor effect.

So the difference between the two tests is one boolean flag plus the target's seed source. One
harness with two toggles covers both.

A property worth stating, because it is a built-in sanity check: the argmax can only *suppress* a
non-winning object, never raise it. So under Test 1 the target can only **lose** pixels to a
neighbor, never gain them. A correct Test 1 diff shows losses only. Any gain in Test 1 is a bug,
not a finding.

## Verified API facts (checked against the installed source and a live napari 0.7.0)

SAM2 (`sam2_video_predictor.py`, `sam2_base.py`):

- Multi-object tracking is just repeated `add_new_points_or_box` / `add_new_mask` calls with
  distinct `obj_id` on one `inference_state`; objects auto-register.
- `add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels, clear_old_points,
  box)`: a box requires `clear_old_points=True` (the box must precede any point), but box plus a
  positive point combine in a single call. Coordinates are in video-frame pixels.
- `add_new_mask(inference_state, frame_idx, obj_id, mask)`: a 2D bool mask, auto-resized to the
  model's image size.
- `propagate_in_video(inference_state, start_frame_idx, max_frame_num_to_track, reverse)` yields
  `(frame_idx, obj_ids, video_res_masks)` covering **all** objects at once; each object's slice
  is shaped `(1, H, W)` and must be squeezed.
- `non_overlap_masks` and `non_overlap_masks_for_mem_enc` are plain attributes on the predictor,
  settable at runtime and restored after use.
- The constraint is a **no-op when only one object is tracked** (`batch_size == 1` returns the
  masks unchanged). So the experiment is meaningless with zero neighbors, and the treatment must
  refuse to run until at least one neighbor is seeded.

napari 0.7.0:

- `add_image` / `add_labels` / `add_points`; a 3D array drives the z-slider via
  `viewer.dims.current_step`.
- Click-to-seed uses `layer.mouse_drag_callbacks` (a list on the layer instance) plus
  `layer.world_to_data(event.position)` to turn a click into `(z, y, x)`.
- Labels `paint` mode supports correcting the target mask; `Points.add` shows seeds
  programmatically.

## Architecture

Everything lives in `coprop_lab.py` at the repo root. It imports the pipeline *library* for the
science (frame loading, prompt building, image-mode prediction, state deserialization) and napari
for the viewer, but imports none of the drivers and none of `gui.py`. The pipeline library is not
modified.

Four parts, each with one job:

### 1. Chain loader (`load_chain(neuron, chain_idx) -> LabChain`)

Reads an already-run chain from disk: the prepared propagation frames (the exact `_sam` or
`_pcrop` images production propagated over) and its `state.json`. Returns a small `LabChain`
record: the frame stack as a `(Z, H, W)` array for display, `frames_dir` (the path SAM2's
`init_state` needs), `anchor_idx`, target `obj_id`, the saved target prompts, the saved anchor
mask (for the correct-seed option), and `frame_to_z`. Pure I/O and array assembly, no torch.

### 2. `MultiObjectCopropSession` (the only torch-touching part)

A small class ported from the keeper on the reverted branch, defined in `coprop_lab.py` (not added
to `pipeline/`). It is deliberately separate from `PropagationSession`, which is single-object and
production-coupled.

- Holds one `inference_state` over the chain's `frames_dir`.
- `seed_target(spec)`: seeds object 1 on the anchor frame, either from the saved prompts
  (box plus positive, `clear_old_points=True`) or from a 2D mask via `add_new_mask`.
- `seed_neighbor(obj_id, box, points, labels)`: seeds a neighbor object on the same anchor frame
  from a box plus point produced by the click handler.
- `run(variant) -> {obj_id: {frame_idx: bool mask}}`: sets the predictor flag for the chosen
  variant (`none`, `output_only`, or `memory`), propagates bidirectionally from the anchor,
  collects every object's mask per frame (squeezing the `(1, H, W)` slice), and restores both
  flags on exit, so a shared predictor is never left mutated.
- Baseline and treatment use **separate `inference_state`s** with identical seeds, so the two
  passes cannot contaminate each other's memory. For a test, two clean passes is the
  unambiguous choice over reusing and resetting one state.

The class uses SAM2's own flags rather than reimplementing the argmax, so what it measures is
exactly what the model does.

### 3. Neighbor seeding by click

A `mouse_drag_callbacks` handler on the EM layer. A click at the anchor frame:

1. turns the click into `(z, y, x)` via `world_to_data`,
2. runs the pipeline's existing image-mode predict (`pipeline.image_predict`) at that point for a
   new object, deriving a mask and a box,
3. shows the resulting neighbor mask immediately as a layer, so the user confirms it grabbed the
   right cell **before** committing,
4. records the neighbor's `(box, point)` seed keyed by a fresh `obj_id`.

A "remove last neighbor" control drops a wrong grab. This is the direct fix for the reverted
version's blind, wrong-cell seeding: every neighbor is seen and approved before any propagation.
There is no CATMAID neighbor lookup and no cross-z anchor mapping; the user decides which neighbors
matter and clicks them, all on the single anchor frame, in the one propagation space.

### 4. The viewer app (`CopropLab`)

Builds the napari viewer and wires the controls:

- Base EM layer (the `(Z, H, W)` stack), parked on the anchor frame.
- A target paint (Labels) layer, preloaded with the saved anchor mask; editable for the
  correct-seed path.
- A neighbors layer (the clicked neighbor masks, distinct colors) and a seeds Points layer.
- Controls: a target-seed radio (`auto-prompts` or `current mask`), a non-overlap variant radio
  (`output-only` or `memory`), a neighbor count readout, and the two preset buttons:
  - **Test 1**: target seed = auto-prompts, variant = output-only.
  - **Test 2**: target seed = current mask, variant = memory.
- A "run A/B" button that runs the baseline pass (flags off) then the treatment pass (chosen
  variant), guarded to refuse when no neighbor is seeded.
- Result layers after a run: `target (alone)`, `target (w/ neighbors)`, `neighbors`, and a `diff`
  layer (lost in one color, gained in another), plus a small text readout of per-frame and total
  pixels gained and lost.

The comparison is the user scrubbing z and toggling layer visibility, documented with napari
screenshots. Nothing is written to disk.

## Data flow

```
load_chain(neuron, chain)                  # frames + state.json from disk
        |
napari: EM stack + target paint (saved anchor mask) at anchor frame
        |
click neighbors  -> pipeline.image_predict -> show neighbor mask -> record (box, point)
        |
choose target seed (auto-prompts | current mask) and variant (output-only | memory)
        |
run A/B:
  Pass A  MultiObjectCopropSession(variant="none")    -> target (alone),    neighbors
  Pass B  MultiObjectCopropSession(variant=chosen)    -> target (w/ neighbors)
        |
build diff (alone vs with-neighbors) + pixel readout
        |
napari layers: EM | target (alone) | target (w/ neighbors) | neighbors | diff
```

## Tests

The pure helpers are torch-free and napari-free, so they get CPU unit tests in
`tests/test_coprop_lab.py`, ported from the reverted branch's `test_gui_coprop_layers.py`:

- `neighbor_label_stack`: paints each neighbor object's mask into a `(Z, H, W)` label stack,
  squeezing the `(1, H, W)` SAM2 slice, and never paints the target id.
- `build_diff_stack`: marks lost and gained target pixels between the two passes.

The session is torch-bound and is exercised by hand through the GUI, matching the repo's existing
test split (CPU-only suite, GPU paths run manually).

## How to run

```
py -3 coprop_lab.py --neuron AVAL --chain 7
```

Best run on a tier-2 `_pcrop` chain: the window is smaller and higher-resolution, which fits the
RTX 3050's 6 GB with a few neighbors. The reverted run already showed three neighbors on a
`_pcrop` window is feasible.

## Risks and caveats

- The memory variant (Test 2) changes the trajectory, so a neighbor with a poor mask can pull
  real target pixels away; the reverted run saw exactly this drift with auto-selected neighbors.
  Manual placement plus the immediate neighbor-mask preview is the mitigation: the user only
  commits neighbors whose masks look right.
- Coordinate space is the usual hazard. All seeds are placed on the single anchor frame in the
  chain's own propagation space (`_sam` or `_pcrop`), so there is no cross-z or cross-space
  mapping to drift.
- VRAM scales with object count. Default to a small number of neighbors and prefer `_pcrop`
  chains. The image encoder runs once per frame regardless; the per-object cost is the memory
  attention and bank.

## Out of scope

Ground-truth scoring, batch runs, any change to `run_chain`, `PropagationSession`, the production
`gui.py`, or the pipeline library, and any persistence of results. If a test is promising, a
GT-scored version is a separate, later piece of work.
