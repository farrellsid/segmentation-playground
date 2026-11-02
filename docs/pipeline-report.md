# Segmentation Pipeline Documentation

This pipeline segments the *C. elegans* connectome: about 300 neurons, a few thousand
maximal-linear-chains, out of a roughly 300-slice electron-microscopy stack, into per-neuron 3D mask
volumes for export to Blender. It is built on the same SAM2 image-mode-to-video-mode propagation
approach as the Cheng-Bader liver vEM pipeline, and it keeps that backbone: skeleton prompt, image
prediction, bounding box, bidirectional video propagation, instance masks, Blender.

The point of this document is to record what the neuron morphology forced me to change relative to
that liver approach. These are adaptations to a different kind of object, not claims that any part is
better. Liver structures are compact, convex, sparsely packed, and few. Neurons are thin, densely
packed against their neighbours, numerous, and branched. Almost every departure below traces back to
one of those differences. (Figure B contrasts the two regimes; Figure A is the stage-by-stage diff.)

Repository: https://github.com/farrellsid/segmentation-playground

## Current pipeline

### Inputs

Two inputs go in:

- **Raw EM images**, a `.tif` stack (about 9000 x 9000 px per slice, roughly 300 slices).
- **CATMAID skeleton annotations**, pulled from the CATMAID server through its API and processed into
  a node table (`aggregate_data_pv.csv`) plus a chain decomposition (`chains.json` / `roots.json`).
  These are hand-traced centerlines, so they give the route through each neuron, not its outline on
  any slice.

### Skeleton to linear chains (because neurons branch)

The skeleton is processed in two steps before any segmentation:

1. **Virtual nodes** are inserted so the centerline is sampled densely enough along z, bridging gaps
   up to a set section allowance.
2. **Branch detection by DFS** walks the tree and splits it into maximal linear chains. A branch is a
   node with multiple children, or a merge where multiple parents converge on one child. The rule is
   a heuristic and does not catch every case, but it is right for the large majority.

The reason this step exists: SAM2's video memory tracks a single object and, at a branch, keeps
following one arm. A whole branching neuron cannot propagate as one object, so it has to be cut into
single-object chains first, propagated separately, and recomposited into the per-neuron volume. The
liver pipeline never needs this step because each of its structures is one compact object. (Figure C.)

### Segmentation in two resolution passes (because neurites are thin and frames are large)

Video propagation runs at an 8x downscale, which is the canonical space the masks are stored in. Two
things make that not enough on its own:

- A neurite is only a few pixels wide at 8x, so the anchor is predicted on a higher-resolution crop
  taken around the skeleton node, and the resulting box is mapped back to seed propagation.
- There is an optional per-chain crop (tier 2): crop a window sized to the whole chain's skeleton
  extent and run the entire image and propagation pass inside it at about a 2x downscale. The window
  is read coarser when it would otherwise be too large, so memory stays bounded.

The memory limit is real: feeding a full 9000 x 9000 frame to SAM2 at native resolution will exhaust
the memory on a normal GPU and crash. The liver pipeline uses a single fixed crop size; the worm
needs adaptive, chain-sized crops because neurite extent varies widely and the thin cross-sections
need the resolution. With Compute Canada access now available, the heavy passes may be able to run at
full resolution, leaving the cropping for the user-facing GUI.

### SAM2 prompting (because neighbours are packed tightly)

The core prediction follows the Cheng-Bader liver approach. Image mode is seeded on the anchor slice
with the skeleton node as a positive point, plus up to seven negative points placed on the
neighbouring neurons around it. That produces an anchor mask, which (with its bounding box) seeds
video mode, and the mask is propagated forward and backward through z.

The negative points are the adaptation. Neurites press right up against their neighbours, so the
model has to be told which cross-section is the target and which are not, or the mask bleeds into an
adjacent cell. In the sparse liver case a single positive point is enough.

### Automation and review

The batch runner processes every chain headless, usually overnight. As it runs it scores each frame
with automatic quality checks (mask-area ratio, temporal IoU against the neighbouring slice, skeleton
containment, and SAM2's own predicted-IoU) and flags the frames that look wrong. A chain is queued
for a human when enough of its frames flag.

During work hours the annotator opens the results and corrects the flagged chains in a napari GUI,
chain by chain. Once a frame has been checked by hand, that corrected mask can be passed directly to
the video propagator as the seed, which is a stronger form of conditioning than points or a box,
since a verified mask is the highest level of confidence available.

The reason for the QC-and-triage layer: thousands of chains rule out reviewing everything by hand,
and the harder morphology means more chains need a human than in the liver case, so the automatic
checks exist to spend the annotator's time only on the flagged fraction. (Figure D.)

## Developer documentation

More detailed, code-level documentation for future users and developers (example usage, parameters,
which file to run for which job) lives in the repository under `docs/` (a getting-started tutorial,
how-to guides, a code map, and configuration reference) and in the module docstrings.

## Future work

- **Multi-instance segmentation**, segmenting a chain's neighbours alongside it for extra context.
- **Per-frame prompting.** Caveat: this needs virtual nodes on every frame, and some currently appear
  to be missing, which still needs to be verified.
- **Integrating a U-Net or Cheng's foundational model** into parts of the pipeline where a trained,
  EM-specialised model may do better than prompted SAM2.
- **Finetuning and parameter optimisation of SAM2** (further out). The cross-worm ground truth now in
  hand makes both of these possible: finetuning has a training target, and parameters can be tuned
  against a real accuracy metric rather than by feel.

## Tried and shelved

Approaches that were tried and set aside, with the reason each was dropped. (To fill in.)
