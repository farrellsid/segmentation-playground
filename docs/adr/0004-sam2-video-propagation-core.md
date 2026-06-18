# 0004. SAM2 video propagation as the core

Status: Accepted

## Context

The task is to segment one neuron at a time through about 300 z-slices. The expensive part is the
per-slice propagation, not the one-time anchor on the mid-slice. Two facts shaped the choice of
method. The professor directed the project to use SAM2. There was direct prior art in the same lab
ecosystem: the Bader Lab human-liver pipeline and its sam2maskpropagator code, which does almost
exactly this (a skeleton point prompt, image predict, largest connected component, bounding box,
then bidirectional propagation to instance masks for Blender).

## Decision

Use SAM2's video predictor as the core. For each chain: take a CATMAID skeleton node as the anchor on
its mid-slice, prompt SAM2 image mode to get a mask and a box, seed the video predictor with that box
on the anchor frame, and propagate the mask bidirectionally through z.

## Consequences

- A promptable foundation model needs no training to stand up. The existing CATMAID skeletons are the
  prompts.
- The video memory propagates one object across the stack cheaply, which is exactly the expensive
  part being automated.
- The known limits are accepted as the starting point, not the ceiling. SAM2 downsamples away fine
  detail that thin neurites depend on, its memory assumes smooth motion that branch points violate,
  and it carries natural-image priors rather than EM-adapted ones. The crop tiers
  ([0009](0009-tier2-crop-fallback.md)) and the finetuning and dense-segmentation directions in the
  [roadmap](../explanation/roadmap.md) exist to address these.
- The base-paradigm question stays open and evidence-gated: if a trained dense model clearly beats
  finetuned SAM2 on the ground truth, the SAM2-as-core directive is something to revisit with the
  professor, with ERL numbers rather than argument.
