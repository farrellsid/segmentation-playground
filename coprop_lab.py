"""Co-propagation lab: a standalone, disposable napari test for the neighbor-competition
hypothesis. Not part of the pipeline; saves nothing, scores nothing. See
docs/superpowers/specs/2026-06-30-coprop-lab-design.md.

torch and napari are imported lazily inside the functions that need them, so importing
this module for the pure helpers stays CPU-only.
"""

from __future__ import annotations

import numpy as np


def label_stack(segments, obj_ids, t, hw):
    """(t, H, W) uint8: each obj_id in `obj_ids` painted with its own id, background 0.

    `segments` is {frame_idx: {obj_id: mask}} where a mask is bool HxW or the SAM2
    video shape (1, H, W); the leading dim is squeezed. Used for the target-alone, the
    target-with-neighbors, and the neighbors layers, by passing the relevant id list.
    """
    H, W = hw
    out = np.zeros((t, H, W), dtype=np.uint8)
    for fi, seg in segments.items():
        if not (0 <= fi < t):
            continue
        for oid in obj_ids:
            if oid in seg:
                m = np.asarray(seg[oid])
                m = m[0] if m.ndim == 3 else m   # SAM2 logits carry a leading (1,H,W) dim
                if m.shape == (H, W):
                    out[fi][m.astype(bool)] = oid
    return out


def build_diff_stack(alone, withn, obj_id):
    """(t, H, W) uint8: 1 where the target had a pixel ALONE but lost it with neighbors
    (bleed carved out), 2 where it GAINED one. The visual read of the test. Under the
    output-only variant the target can only lose pixels, so a correct Test 1 diff has no 2s.
    """
    a = (alone == obj_id)
    b = (withn == obj_id)
    out = np.zeros_like(alone, dtype=np.uint8)
    out[a & ~b] = 1     # lost (carved-out bleed)
    out[b & ~a] = 2     # gained
    return out
