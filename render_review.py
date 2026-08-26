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


from sam2_utils import bundle as bundle_utils
from sam2_utils import meshing

#: Nanometres per voxel at `_sam` scale 8, as (z, y, x). Full res is 16nm in xy and
#: 50nm in z, so xy coarsens by 8 to 128nm while z does not change. z is therefore the
#: FINER axis, which is the opposite of the usual EM intuition.
SPACING_SAM8_NM = (50.0, 128.0, 128.0)


def _chain_dirs(root: Path, neuron: str):
    """Every chain directory for one neuron, in chain order, bundle or tree alike."""
    recs = bundle_utils.index_chains(Path(root), neurons=[neuron])
    return [(int(r["chain_idx"]), Path(root) / r["chain_dir"])
            for r in sorted(recs, key=lambda r: int(r["chain_idx"]))]


def neuron_volume(root, neuron: str, *, max_voxels: int = 400_000_000):
    """Every chain of ``neuron`` composited onto one grid, cropped to its bbox.

    Cropping is what keeps this laptop-sized. The full `_sam` grid across a long
    neuron runs to hundreds of MB, while the neuron's own bounding box is a small
    fraction of that. Returns ``(vol, spacing)`` with ``vol`` as ``(z, y, x)`` uint8.
    """
    root = Path(root)
    blocks = []
    for _ci, cdir in _chain_dirs(root, neuron):
        for z, (mask, x0, y0) in pipeline.chain_masks_in_sam(cdir).items():
            if mask.any():
                blocks.append((int(z), np.asarray(mask, dtype=bool), int(x0), int(y0)))
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.uint8), SPACING_SAM8_NM

    zs = sorted({b[0] for b in blocks})
    z_index = {z: i for i, z in enumerate(zs)}
    x0 = min(b[2] for b in blocks)
    y0 = min(b[3] for b in blocks)
    x1 = max(b[2] + b[1].shape[1] for b in blocks)
    y1 = max(b[3] + b[1].shape[0] for b in blocks)
    shape = (len(zs), y1 - y0, x1 - x0)
    n_vox = shape[0] * shape[1] * shape[2]
    if n_vox > max_voxels:
        raise SystemExit(
            f"[render] {neuron} needs a {shape} volume, {n_vox:,} voxels, over the "
            f"{max_voxels:,} budget. Render fewer chains, or raise --max-voxels if "
            f"this machine has the memory.")

    vol = np.zeros(shape, dtype=np.uint8)
    for z, mask, bx, by in blocks:
        h, w = mask.shape
        sy, sx = by - y0, bx - x0
        vol[z_index[z], sy:sy + h, sx:sx + w] |= mask.astype(np.uint8)
    return vol, SPACING_SAM8_NM


def neuron_mesh(root, neuron: str, out_path, *, preset: str = "faithful",
                max_voxels: int = 400_000_000):
    """Write one PLY for ``neuron``. Returns the path, or None if it has no masks."""
    vol, spacing = neuron_volume(root, neuron, max_voxels=max_voxels)
    if not vol.size or not vol.any():
        print(f"[render] {neuron}: no masks, skipping mesh")
        return None
    verts, faces = meshing.volume_to_mesh(vol, spacing=spacing, preset=preset)
    if len(faces) == 0:
        print(f"[render] {neuron}: no surface produced, skipping mesh")
        return None
    out = meshing.write_ply(out_path, verts, faces)
    print(f"[render] {neuron}: mesh {len(verts):,} verts, {len(faces):,} faces -> {out}")
    return out
