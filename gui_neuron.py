"""gui_neuron.py: napari NEURON-level review GUI (the second review paradigm).

The per-chain tool (gui.py) opens one chain at a time. This one opens a whole NEURON:
all its chains (branches) on a single per-neuron crop canvas (_ncrop), shown as one
multi-color object. Branches stay separate SAM2 objects; the neuron is a presentation +
union layer. See docs/superpowers/specs/2026-06-23-neuron-review-gui-design.md and the
plan docs/superpowers/plans/2026-06-23-neuron-review-gui.md.

gui.py is untouched; this driver imports its shared pieces (ReviewContext, helpers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sam2_utils import review_queue


def neurons_on_disk(output_root) -> list[tuple[str, list[int]]]:
    """Every neuron with on-disk chains under output_root, as (neuron, [chain_idx,...]),
    sorted by neuron then chain. Built on ReviewQueue.all_chains so the openable set
    matches exactly what review.load_chain can read."""
    q = review_queue.ReviewQueue(Path(output_root))
    by_neuron: dict[str, list[int]] = {}
    for neuron, idx in q.all_chains():
        by_neuron.setdefault(neuron, []).append(idx)
    return [(n, sorted(by_neuron[n])) for n in sorted(by_neuron)]


def build_neuron_label_volume(branch_masks: dict, t: int,
                              hw: tuple[int, int]) -> np.ndarray:
    """(t, H, W) uint16 volume: for each branch label L and frame fi,
    branch_masks[L][fi] (bool, shape hw) is written as L. Ascending label order, so a
    higher label wins on overlap (deterministic). Labels are the per-branch editing
    integers; the saved neuron identity is independent of them."""
    H, W = hw
    vol = np.zeros((t, H, W), dtype=np.uint16)
    for label in sorted(branch_masks):
        for fi, m in branch_masks[label].items():
            if 0 <= fi < t:
                vol[fi][np.asarray(m, bool)] = label
    return vol
