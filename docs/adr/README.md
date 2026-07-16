# Architecture decision records

An ADR is a short document that captures one decision: the context it was made in, the decision
itself, and the consequences. The format follows Michael Nygard's template.

The point of these is the newcomer problem. Someone arriving on the project will hit a past decision
and be unsure whether to accept it or change it. Without the reasoning, they can only guess. These
records hold the reasoning.

Rules:

- One decision per file, numbered in order.
- Never edit a decision after it lands. If it changes, write a new ADR and mark the old one
  superseded, with a link.
- Record load-bearing decisions, not trivia.

For the running build history (what changed and when), see [../CHANGELOG.md](../CHANGELOG.md).

## Index

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-library-plus-thin-drivers.md) | Library plus thin drivers | Accepted |
| [0002](0002-serializable-chain-state.md) | Serializable per-chain state | Accepted |
| [0003](0003-filesystem-only-no-database.md) | Filesystem only, no database | Accepted |
| [0004](0004-sam2-video-propagation-core.md) | SAM2 video propagation as the core | Accepted |
| [0005](0005-centralized-coordinate-transforms.md) | Centralized coordinate transforms | Accepted |
| [0006](0006-canonical-mask-space.md) | Canonical mask space and encoding | Accepted |
| [0007](0007-napari-review-gui.md) | napari for the review GUI | Accepted |
| [0008](0008-video-seed-box-vs-mask.md) | Box-plus-point seed for auto, mask for the GUI | Accepted |
| [0009](0009-tier2-crop-fallback.md) | Tier-2 per-chain crop with image-score fallback | Accepted |
| [0010](0010-erl-voi-eval-ruler.md) | ERL and split/merge VOI as the eval ruler | Accepted |
| [0011](0011-flat-layout-over-src.md) | Flat layout over src layout | Accepted |
| [0012](0012-node-anchored-multimask-selection.md) | Node-anchored multimask selection | Accepted |
| [0013](0013-pipeline-package-split.md) | The pipeline core is a package split by concern | Accepted |
| [0014](0014-neuron-level-review-gui.md) | Neuron-level review as a second GUI paradigm | Accepted |
| [0015](0015-target-worm-merge-metric-ruler.md) | Target-worm skeleton merge-metric as the GT-free bleed ruler | Accepted |
