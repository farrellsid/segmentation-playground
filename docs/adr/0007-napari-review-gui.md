# 0007. napari for the review GUI

Status: Accepted

## Context

QC flags chains that need a human, so the project needs one tool where a reviewer can scrub a chain,
inspect why it flagged, edit prompts, paint a correction, re-run the anchor, and resume propagation.
A bad anchor and a mid-propagation drift are the same kind of problem (a frame that needs a prompt
edit), so they should share one tool, not two. The earlier matplotlib-widget prototypes were
unresponsive in Jupyter and were never going to carry an interactive correction loop.

micro_sam ships a napari plugin for interactive EM segmentation, which raised a build-versus-adopt
question.

## Decision

Build the review GUI on napari (`gui.py`), composing the existing library: the read-only viewer to
rebuild a chain's overlay, the review queue, the label store, and the propagation primitives for
re-segmentation. Everything the GUI shows lives on one `_sam` grid, so a click is already a grid
coordinate. The GPU is lazy: browsing, scrubbing, painting, and labeling need no predictors; the SAM2
models are built only on the first re-run.

## Consequences

- One tool covers anchor review and drift correction, against one triage queue.
- A corrected chain is rewritten on disk (masks, `qc.csv`, `state.json`) so it is indistinguishable
  from a fresh batch run.
- The GUI collects per-frame labels as a side effect of review, which is the training data for a
  future learned QC detector.
- micro_sam's plugin stays a build-versus-adopt option to revisit, independent of any model swap.
  This ADR is the build path.
- Concurrent reviewers are out of scope for now (no cross-process lock). See
  [0003](0003-filesystem-only-no-database.md).
