"""Render a reviewed neuron as a video and as a Blender-ready mesh.

Runs against a review bundle or an output tree. Both are laid out
``<neuron>/chain_NN/state.json``, so `bundle.index_chains` indexes either and
`pipeline.chain_masks_in_sam` reads masks from either. They differ in exactly one way
that matters here: a bundle carries its own per-chain ``frames/`` while a tree expects
the EM store. That difference lives in `chain_frames` and nowhere else, which is what
lets this run on a reviewer's laptop with no EM store and no torch.

    py -3 render_review.py --source ~/mask-review/AIB --out AIB/review
    py -3 render_review.py --gui
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline


def source_kind(root) -> str:
    """``"bundle"`` or ``"tree"``, decided by which marker file is present."""
    root = Path(root)
    if (root / "bundle.json").exists():
        return "bundle"
    if (root / "_manifest.csv").exists():
        return "tree"
    raise SystemExit(
        f"[render] {root} is neither a bundle nor an output tree: no bundle.json and "
        f"no _manifest.csv. Point --source at a bundle folder or a tree root.")


def chain_frames(chain_dir, state: dict, kind: str) -> Dict[int, np.ndarray]:
    """``{frame_idx: RGB uint8}`` for one chain, from wherever this source keeps EM.

    A bundle reads the JPEGs it shipped with, which is what makes this work on a
    machine that has never seen the raw stack. A tree reads the EM store at each of the
    chain's own z values.
    """
    chain_dir = Path(chain_dir)
    if kind == "bundle":
        fdir = chain_dir / "frames"
        paths = sorted(fdir.glob("*.jpg")) if fdir.is_dir() else []
        if not paths:
            raise SystemExit(
                f"[render] {chain_dir.parent.name}/{chain_dir.name} has no frames/ in "
                f"this bundle, so its EM cannot be drawn. Re-export the bundle rather "
                f"than rendering a neuron with a chain silently missing.")
        out = {}
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                raise SystemExit(f"[render] could not read {p}")
            out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return out

    frame_to_z = {int(k): int(v) for k, v in (state.get("frame_to_z") or {}).items()}
    out = {}
    for fi in sorted(frame_to_z):
        em, _full = pipeline.load_frame_sam(frame_to_z[fi], scale=8)
        out[fi] = em if em.ndim == 3 else np.stack([em] * 3, axis=-1)
    return out
