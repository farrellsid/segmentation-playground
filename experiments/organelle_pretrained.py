"""Full-frame organelle mask from pretrained detectors (MitoNet + NucleoNet), an alternative
mask source for sam2_utils.membrane.suppress_organelles besides detect_organelle_blobs's
shape/intensity heuristic.

Both nuclei and mitochondria corrupt the membrane ridge map (roadmap items 2e/2f): the
classical heuristic calibrated to ~96% single-pixel Otsu noise on real EM and moved the
real bleed-count floor by zero (see docs/CHANGELOG.md's 2026-07-29 organelle blob
suppression entry). This unions MitoNet's and NucleoNet's real instance-segmentation output
into one boolean mask instead, dilated the same way detect_organelle_blobs dilates its own
output, so a caller like dense_membrane_fill.py's grow_all does not need to know which
detector produced the mask it is inpainting out.

Lazy-imports torch/empanada via the two spotcheck modules, so importing this module alone
does not pull in either heavy dependency until detect_organelles_pretrained actually runs.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def detect_organelles_pretrained(em_gray: np.ndarray, *, include_nuclei: bool = True,
                                  dilate_px: int = 1) -> np.ndarray:
    """Run MitoNet (always) and NucleoNet (if include_nuclei) on a full grayscale EM frame,
    return the union of both instance-segmentation outputs as one dilated boolean mask,
    same H x W as em_gray.

    Two full-frame forward passes total, not one per neuron crop: run this once per frame
    and have callers slice the region they need out of the returned mask, the same pattern
    dense_membrane_fill.py already uses for frames (a {z: em_gray} dict built once, sliced
    per neuron)."""
    from experiments.mitonet_spotcheck import run_mitonet
    from experiments.nucleonet_spotcheck import run_nucleonet

    out = run_mitonet(em_gray) > 0
    if include_nuclei:
        out = out | (run_nucleonet(em_gray) > 0)
    if dilate_px > 0 and out.any():
        out = ndi.binary_dilation(out, iterations=dilate_px)
    return out
