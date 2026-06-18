# data/groundtruth

**Status: git-ignored data landing spot.** The cross-worm ground-truth dataset lands here; only this
README is tracked.

**This directory is the landing spot for the cross-worm ground-truth dataset.** The data itself is
**not committed**, `data/` is git-ignored; only this README is tracked (via a `.gitignore`
exception) so the scaffold is visible.

Per [`../../docs/explanation/roadmap.md`](../../docs/explanation/roadmap.md) §3, the June 2026 step-back obtained
ground-truth segmentation for **a different worm**, a separate EM stack with a somewhat different
look, *with the matching EM images and explicit markers for which segments are manually confirmed*.
This is the pivotal unlock that turns the previously "label-gated" M4.5 backlog into real
**evaluation** ([`../../eval/`](../../eval/), Stage 0) and **finetuning**
([`../../finetune/`](../../finetune/), Stage 2).

## What lands here (when exported)

- **VAST-Lite export** of the GT segmentation (the manual-segmentation source), plus
- the **matching EM** image stack, and
- the **confirmed-segment markers**, the per-segment flags saying which segments are manually
  verified. Stage 0/Stage 2 must **supervise / score only on confirmed voxels** (FUTURE_DIRECTIONS
  §4.7, §3 caveat).

## Caveat (read before trusting any number)

Cross-specimen GT measures **generalization**, not in-distribution accuracy, treat it as a
domain-adaptation benchmark and spot-check on the *target* worm (FUTURE_DIRECTIONS §3, §7).
