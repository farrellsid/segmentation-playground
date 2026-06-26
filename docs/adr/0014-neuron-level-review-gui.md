# 0014. Neuron-level review as a second GUI paradigm

Status: Accepted

## Context

The pipeline decomposes a neuron's CATMAID skeleton into linear chains and tracks each
independently, so the per-chain GUI (`gui.py`) reviews one chain at a time. The annotator
thinks in whole neurons ("click a neuron, see it traced, fix it"), and some chains are
artificial splits of one continuous neurite, which fragments a neuron's identity across
several review units. A whole-neuron review tool was wanted, without disturbing the
working per-chain pipeline.

The obvious approach, track the whole neuron as one SAM2 object, was rejected after
checking the SAM2 source: a video object is tracked through a single z-ordered sequence
with its own memory bank, and there is no cross-object attention. Asking one object to
carry the multiple cross-sections a branch point produces invites the memory to drop or
swap branches. A neuron's arbor is also a branching tree spread non-monotonically over z,
which a single sequential track does not represent.

## Decision

Add `gui_neuron.py` as a separate driver, a second review paradigm, leaving `gui.py` and
the batch untouched. Keep each branch a separate SAM2 object (the per-chain tracking that
already works); the neuron is a presentation and union layer on top:

- One per-neuron crop window (`_ncrop`), sized to the whole neuron's skeleton extent, is
  the single canvas every branch remaps into, so a neuron's mixed `_sam` / `_pcrop`
  branches share one space.
- One napari Labels layer holds an integer per branch; the selected label is the active
  branch. Corrections (re-predict, resume, prompts, box) act on the active branch over a
  z-scoped view of its own frames, and write back to its label.
- The multi-blob neuron result on a branch-point slice is produced by union, not by
  tracking, so it gets the joint appearance without the single-object failure mode.

The driver reuses `gui.py`'s shared helpers (driver to driver), which is allowed by the
library/driver rule in [0001](0001-library-plus-thin-drivers.md).

## Consequences

- Two review tools coexist: `gui.py` per chain, `gui_neuron.py` per neuron. They share the
  same correction primitives and on-disk format; a branch edited in either is just a chain
  whose `crop_window` records its space ([0006](0006-canonical-mask-space.md)).
- The `_ncrop` resolution adapts to neuron size: a sprawling neuron gets a coarser canvas
  (toward `_sam`), so the per-chain GUI remains the place to fix one thin branch at full
  tier-2 sharpness.
- Cross-neuron overlap arbitration (the ERL win) is deferred; within a neuron, branches
  union with no conflict.
- Multi-instance tracking gives non-overlap only via an optional post-hoc argmax, not
  learned context, so it is not the headline win; the genuine non-overlap-plus-no-underfill
  route (segment-everything-then-associate) remains future work.

See the design spec and plan under `docs/superpowers/` (2026-06-23).
