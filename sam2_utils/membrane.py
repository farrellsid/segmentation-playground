"""Membrane / boundary signal for the target worm (roadmap Phase 2, foundation).

A ground-truth-free per-pixel membrane-ness map read from the raw EM, plus the
pure detector primitives that grade a mask against it. The map generator is v1
(a classical dark-ridge filter); the signature is the interface, so a trained
model can drop in behind membrane_map() later without touching the detectors or
the eval scorer. Design:
docs/superpowers/specs/2026-07-17-phase2-membrane-map-bleed-detection-design.md
"""
from __future__ import annotations

import numpy as np

# v1 defaults, resolution-aware for the _sam grid (scale ~8). Comparative, not absolute.
DEFAULT_SIGMAS = (1, 2, 3)
DEFAULT_TAU = 0.5   # membrane threshold on the normalised [0, 1] map
DEFAULT_F = 0.15    # min component area as a fraction of the mask, for spanning
DEFAULT_TOL = 2     # px tolerance for boundary-on-membrane
DEFAULT_K = 6       # px flood radius for underfill


def membrane_map(em_patch: np.ndarray, *, sigmas=DEFAULT_SIGMAS) -> np.ndarray:
    """Per-pixel membrane-ness in [0, 1] for a grayscale or RGB EM patch.

    v1: a Sato dark-ridge filter (membranes are dark on bright cytoplasm),
    normalised by its 99th percentile so tau is stable across frames. Returns
    float32, same H x W as the input.
    """
    from skimage.filters import sato

    img = em_patch
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    resp = sato(img, sigmas=sigmas, black_ridges=True).astype(np.float32)
    denom = float(np.percentile(resp, 99)) + 1e-6
    return np.clip(resp / denom, 0.0, 1.0).astype(np.float32)
