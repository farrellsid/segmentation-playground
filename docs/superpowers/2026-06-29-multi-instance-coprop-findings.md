# Multi-instance co-propagation: findings and errors (for future chats)

Date: 2026-06-29
Status: feature REVERTED off the working branch. The implementation is preserved on the
branch `feat/multi-instance-coprop` (tip `86a2bf2`) if you want to read or reuse it. The
working branch `repo-reorg` is clean of it (it only carries the spec commit `2f2c396`).

This file is the hand-off note. It records what was built, the bugs that surfaced during
GUI smoke testing, what the real-data runs showed, why the auto-node approach is the wrong
base, and the experiment design to build next (manual node placement, seed-from-correct-mask).

---

## 1. The research question (restated, the version that matters)

Not "how does the default auto-seeded pipeline perform with neighbors." The question is:

> Given a CORRECT first mask (manually placed or verified), how does it propagate? And does
> it propagate BETTER when neighboring cells are co-segmented and propagated alongside it?

Everything below should be read against that goal. The auto-seeding work missed it.

## 2. The mechanism (this part is solid, keep it)

SAM2 multi-object tracking runs the memory attention INDEPENDENTLY per object. One object's
mask never informs another's propagation. The ONLY coupling is
`_apply_non_overlapping_constraints` in `sam2/modeling/sam2_base.py`: a per-pixel argmax over
object scores that keeps the highest-scoring object at each location and suppresses the rest.

Two variants, both plain attributes on the video predictor (settable at runtime, restore after):
- `non_overlap_masks`: argmax on the OUTPUT masks only. Equivalent to a post-hoc labelmap
  argmax. Does NOT change propagation.
- `non_overlap_masks_for_mem_enc`: argmax on the masks FED BACK into memory. This changes what
  each object memorizes, so it alters the propagation trajectory across z. This is the only
  variant that can change tracking, and it is also the one that destabilizes (see section 5).

Consequence: co-propagating neighbors can only change the target through this argmax. With both
flags off, the target mask is bit-identical whether or not neighbors are tracked.

## 3. What was built (on the reverted branch)

- `pipeline.MultiObjectPropagationSession` (pipeline/propagate.py): co-propagate N objects in
  one inference_state, toggles both non-overlap flags with set-and-restore. REUSABLE for the
  manual approach; this is the keeper.
- `pipeline.neighbor_chains(...)` (pipeline/predict.py): auto-find the k nearest other CATMAID
  chains with a node inside the target's window on a shared z. THIS is the part to drop (see 6).
- `pipeline.chain_containing_node(...)`: resolve a chain by a node id (naming-agnostic). Small,
  reusable.
- `ReviewGUI._seed_neighbor` + `coprop_neighbors` (gui.py, key `M`): the auto-node GUI action.
  Ran the session twice (OFF control, ON treatment), added overlay layers `mask (alone)`,
  `mask (w/ neighbors)`, `neighbors`, `diff` (lost=1, gained=2). Later got toggles for
  point-only neighbor seeds and output-only non-overlap.

## 4. Bugs found during smoke testing (all fixed on the branch, listed so they are not re-hit)

1. `self.chain` was None when the on-disk neuron folder name did not match the chains.json
   cell_name (folder "AVAL - Original" vs cell_name "AVAL"). `find_chain` matches the name
   exactly. Fix: resolve the chain by the stored anchor node id (a node belongs to one chain).
2. `neighbor_chains` crashed on `int(node_id)` for CATMAID virtual nodes, which have string ids
   like `v_25425535_1448`. Everywhere else node ids are matched as strings; that int() cast was
   the anomaly. Fix: keep strings verbatim, int() only numerics.
3. `_neighbor_label_stack` indexed a 2D frame with the raw SAM2 `(1, H, W)` per-object logit
   mask. Fix: squeeze the leading dim, same guard as `_label_stack_from_segments`.

Lesson: SAM2 per-object video logits are shaped `(1, H, W)`; every consumer must squeeze. Node
ids are mixed int and `v_*` strings; always compare as strings, never int()-cast.

## 5. What the real runs showed (the behavior that prompted the stop)

Setup: AVAL chain 07 (tier-2, _pcrop window 1360x1272), 3 auto neighbors (RIAL, SAAVL, SMDVL).

- At the seed frame, target-alone and target-with-neighbors look the same. Expected: the
  non-overlap argmax only matters where masks overlap, and at the well-conditioned seed frame
  the target dominates its own region.
- After propagation, the target mask in the ON run JUMPS, onto neighboring nodes or far away.
  Best explanation: `non_overlap_masks_for_mem_enc=True` feeds the argmax-suppressed mask back
  into the target's MEMORY, so every frame a neighbor wins contested pixels the target's
  memorized appearance degrades; over ~17 frames this compounds into drift then a jump. This is
  the known SAM2 long-video error-accumulation mode, amplified by poor neighbor masks.
- It propagates "twice" because the action runs the session OFF then ON, each bidirectional.

Coordinate question (the user asked): the JUMP is almost certainly NOT a coordinate-transform
bug. The OFF run reuses the exact production seed with zero object coupling, so the OFF target
must be pixel-identical to the normal single-instance result; all objects share the same _pcrop
frames and all seeds are produced in that same _pcrop space. The decisive check (not yet run):
toggle `mask (alone)` and confirm it equals the saved production mask. If it matches, coordinates
are proven fine and the drift is purely the mem-enc mechanism.

## 6. Why the auto-node approach is the wrong base (the actual problem in the screenshot)

Observed: clear neighboring cells exist, but the co-propagated neighbor masks highlight the
WRONG cells. Likely causes, in order of suspicion (TO VERIFY, not yet root-caused since we are
reverting):

1. Image-mode seeding of neighbors is unreliable cross-context. `_seed_neighbor` runs
   build_prompts (positive at the neighbor node + nearest-node negatives) then image_predict
   then box_from_mask. Image mode can grab the wrong cell or an oversized region, and the box
   then locks the neighbor object onto the wrong structure.
2. The neighbor's anchor node is the node nearest the target on a SHARED z, but the neighbor is
   then seeded at the neighbor's OWN anchor z and that node is mapped into the target's window.
   On the target's anchor frame the structure at that (x, y) may be a different cell.
3. Possible coordinate concern the user raised (pre-affine vs post-affine). For the TARGET worm
   `catmaid_to_tif` (the fixed stack transform) is the correct map and the per-section
   registration affine is GT-worm-only, so this is the least likely cause, but it has not been
   ruled out by direct check.

Either way, AUTO neighbor selection introduces too many failure modes (which chain, which node,
which z, image-mode mask quality) between the click and the result. It obscures the actual
experiment.

## 7. The other real flaw: the experiment never started from a correct mask

The co-prop action seeds the TARGET from the saved anchor prompts (box + positive point) and
re-propagates from scratch. It does NOT seed from a manually corrected mask. So it can only
report default auto-seed performance, which is not the question.

Note: the single-object `resume_propagation` in gui.py ALREADY mask-seeds (it calls
`session.add_mask(frame_idx, mask)` and propagates from there). The capability exists. The
multi-object path just did not adopt it. The next design should seed the target (and ideally
each neighbor) from a confirmed mask, then propagate.

## 8. Design to build next (the manual approach)

Goal: isolate the propagation question from the seeding noise.

1. Manual placement. Let the user click to place the target seed and each neighbor seed (point,
   box, or a painted mask) directly on the EM canvas. No CATMAID neighbor auto-selection. The
   user decides which neighbors matter and where they are, removing causes 6.1 and 6.2 entirely.
2. Seed the target from a CORRECT first mask. Either the user paints/corrects it, or re-predicts
   and confirms it at the anchor frame, then the session seeds that mask (not the raw prompts).
3. Co-propagate. Run target-alone vs target-plus-neighbors. Keep the OFF/ON structure, and keep
   the output-only vs memory non-overlap distinction (output-only is the gentle coupling that
   does not corrupt memory; memory is the aggressive one that drifts). Compare propagation
   quality frame by frame.
4. Reuse `MultiObjectPropagationSession`, the OFF/ON harness, and the overlay+diff layers. Drop
   `neighbor_chains` and the auto `_seed_neighbor` image-mode path.

Open question worth a cheap test first: does seeding NEIGHBORS from a single manually placed
point (no box) give tighter neighbor masks than the image-mode box did? The branch already has a
"point only" neighbor-seed toggle that hints at this.

## 9. Pointers

- Spec: docs/superpowers/specs/2026-06-26-multi-instance-coprop-design.md
- Plan: docs/superpowers/plans/2026-06-26-multi-instance-coprop.md
- Reverted implementation: branch `feat/multi-instance-coprop`, tip `86a2bf2`
  (12 commits, full CPU test suite green: 190 passed, 1 skipped).
- Progress ledger with the full bug-by-bug history: `.git/sdd/progress.md` on that branch.
- Memory: `multi-instance-coprop-experiment.md`.
