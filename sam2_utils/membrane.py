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
from scipy import ndimage as ndi

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


def register_crops(crops: list[np.ndarray], *, max_shift: int = 5) -> list[np.ndarray]:
    """Align every crop in `crops` to the center crop by an integer-pixel translation.

    Uses phase correlation at pixel precision only, no subpixel interpolation, so no
    blur is introduced by the alignment itself. Each estimated shift is clamped to
    +/- max_shift px per axis before being applied, a safety valve against a
    low-texture crop returning a wild or ambiguous shift. Returns a new list, same
    length and shape as the input; the center crop (index len(crops) // 2) is
    returned unchanged, everything else is float32.
    """
    from skimage.registration import phase_cross_correlation

    n = len(crops)
    if n <= 1:
        return list(crops)
    center_i = n // 2
    ref = crops[center_i].astype(np.float32)
    out = list(crops)
    for i, crop in enumerate(crops):
        if i == center_i:
            continue
        moving = crop.astype(np.float32)
        shift, _error, _diffphase = phase_cross_correlation(ref, moving, upsample_factor=1)
        shift = np.clip(np.round(shift), -max_shift, max_shift)
        out[i] = ndi.shift(moving, shift, order=0, mode="nearest")
    return out


_COMBINERS = {
    "median": lambda stack: np.median(stack, axis=0),
    "mean": lambda stack: np.mean(stack, axis=0),
    "max": lambda stack: np.max(stack, axis=0),
    "min": lambda stack: np.min(stack, axis=0),
}


def project_crops(crops: list[np.ndarray], *, combine: str = "median") -> np.ndarray:
    """Reduce an already-registered window of crops to one projected image.

    `combine` selects the per-pixel reducer across the window: median (default) pulls
    a pixel dark in a minority of slices, a transient organelle, toward the brighter
    majority value, while a pixel dark in most or all slices, a persistent membrane,
    stays dark; mean, max, and min are also available for the same call so a sweep can
    compare them. A single-element `crops` returns that element unchanged for every
    combiner, the natural window=0 fallback.
    """
    if combine not in _COMBINERS:
        raise ValueError(f"unknown combine {combine!r}, choose one of {sorted(_COMBINERS)}")
    stack = np.stack([np.asarray(c, dtype=np.float32) for c in crops], axis=0)
    return _COMBINERS[combine](stack).astype(np.float32)


def _perimeter(mask: np.ndarray) -> np.ndarray:
    """The 1-px inner boundary ring of a boolean mask."""
    return mask & ~ndi.binary_erosion(mask)


def spanning_membrane(mask: np.ndarray, mem: np.ndarray, *,
                      tau: float = DEFAULT_TAU, f: float = DEFAULT_F
                      ) -> tuple[bool, float]:
    """Detect a membrane ridge that spans the mask border-to-border.

    Remove membrane (mem > tau) from the mask, label the remainder, keep
    components with area >= f * area(mask). If two or more kept components each
    touch the mask's outer border, a membrane cut the mask in two: it engulfed a
    cell boundary. Returns (spanning_merge, bled_fraction), bled_fraction being
    the second-largest border-touching component area / mask area.

    A nucleus (a closed interior loop) leaves one border-touching cytoplasm
    region plus one enclosed region that does not touch the border, so a soma is
    not flagged, by construction.
    """
    area = int(mask.sum())
    if area == 0:
        return False, 0.0
    opened = mask & (mem <= tau)
    lbl, n = ndi.label(opened)
    if n == 0:
        return False, 0.0
    perim = _perimeter(mask)
    min_area = f * area
    border_areas: list[int] = []
    for i in range(1, n + 1):
        comp = lbl == i
        a = int(comp.sum())
        if a < min_area:
            continue
        if bool((comp & perim).any()):
            border_areas.append(a)
    border_areas.sort(reverse=True)
    if len(border_areas) >= 2:
        return True, border_areas[1] / area
    return False, 0.0


def boundary_on_membrane(mask: np.ndarray, mem: np.ndarray, *,
                         tau: float = DEFAULT_TAU, tol: int = DEFAULT_TOL) -> float:
    """Fraction of the mask perimeter within tol px of a membrane pixel. Low
    means the edge floats through cytoplasm (leaking bleed or underfill)."""
    perim = _perimeter(mask)
    p = int(perim.sum())
    if p == 0:
        return 0.0
    memb = mem > tau
    if tol > 0:
        memb = ndi.binary_dilation(memb, iterations=tol)
    return float((perim & memb).sum()) / p


def underfill_fraction(mask: np.ndarray, mem: np.ndarray, *,
                       tau: float = DEFAULT_TAU, k: int = DEFAULT_K) -> float:
    """k-bounded flood out of the mask into cytoplasm (mem <= tau), membranes as
    walls. Returns reachable cytoplasm area outside the mask / mask area: high
    means the mask stopped short of its enclosing membrane (room to grow).

    Lowest-confidence of the three detectors: at coarse _sam a broken ridge lets
    the flood leak into a neighbour and overestimate. The k bound keeps a leak
    local. Measured only, never applied (refinement is a separate spec)."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    cyto = mem <= tau
    reach = mask.copy()
    for _ in range(int(k)):
        grown = (ndi.binary_dilation(reach) & cyto) | mask
        if int(grown.sum()) == int(reach.sum()):
            break
        reach = grown
    return float((reach & ~mask).sum()) / area
