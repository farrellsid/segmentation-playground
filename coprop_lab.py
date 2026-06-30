"""Co-propagation lab: a standalone, disposable napari test for the neighbor-competition
hypothesis. Not part of the pipeline; saves nothing, scores nothing. See
docs/superpowers/specs/2026-06-30-coprop-lab-design.md.

torch and napari are imported lazily inside the functions that need them, so importing
this module for the pure helpers stays CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def load_em_stack(frames_dir, n_frames):
    """(n, H, W, 3) uint8 RGB over the chain's 0-indexed frames, eager. Reuses the same
    single-frame reader review/video_viz use, so naming and color match the rest of the repo."""
    from sam2_utils.video_viz import _load_frame
    frames = [_load_frame(Path(frames_dir), i) for i in range(n_frames)]
    return np.stack(frames, axis=0)


@dataclass
class LabChain:
    em: np.ndarray                  # (T, H, W, 3) uint8 RGB, the display stack
    frames_dir: str                 # path SAM2 init_state needs
    anchor_idx: int
    obj_id: int                     # the target object id (== 1 in production)
    frame_to_z: dict
    target_prompts: object          # pipeline.Prompts (box + points in propagation space)
    anchor_mask: np.ndarray         # bool (H, W), the saved target mask at the anchor frame
    n_frames: int
    hw: tuple
    crop_window: Optional[object] = None


def load_lab_chain(output_root, neuron, chain_idx):
    """Build a LabChain from an on-disk, already-run chain. Pure I/O, no torch."""
    import pipeline
    from sam2_utils import review, alignment

    chain_dir = Path(output_root) / neuron / f"chain_{int(chain_idx):02d}"
    if not chain_dir.exists():
        raise FileNotFoundError(f"no chain dir at {chain_dir}")

    data = review.load_chain(chain_dir, verbose=True)        # ReviewData
    t = (max(data.video_segments) + 1) if data.video_segments else 0

    any_mask = next(iter(data.video_segments.values()))[data.obj_id]
    any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
    H, W = np.asarray(any_mask).shape

    anchor_seg = data.video_segments.get(data.anchor_idx, {})
    am = anchor_seg.get(data.obj_id)
    if am is None:
        raise ValueError(f"no target mask at anchor frame {data.anchor_idx}")
    am = np.asarray(am)
    anchor_mask = (am[0] if am.ndim == 3 else am).astype(bool)

    state_path = chain_dir / "state.json"
    state = pipeline.load_state(state_path) if state_path.exists() else None
    prompts = getattr(state, "prompts", None)
    cw = None
    if state is not None and getattr(state, "crop_window", None):
        cw = alignment.CropWindow.from_dict(state.crop_window)

    em = load_em_stack(data.frames_dir, t)
    return LabChain(em=em, frames_dir=str(data.frames_dir), anchor_idx=int(data.anchor_idx),
                    obj_id=int(data.obj_id), frame_to_z=data.frame_to_z,
                    target_prompts=prompts, anchor_mask=anchor_mask,
                    n_frames=t, hw=(H, W), crop_window=cw)
