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
from sam2_utils import bundle as bundle_utils
from sam2_utils import meshing


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

    # The volume spans every z from the minimum to the maximum present, not just the
    # z values that happen to survive the `mask.any()` filter above. Compacting past
    # a missing z (mask dropout is real here, measured at 5-30% over propagation
    # distance) would pull the surviving planes together and silently fuse a real gap
    # to 50nm, which is exactly the defect this mesh exists to reveal. A missing z
    # instead leaves an empty plane, so a real gap in the source data shows up as a
    # real gap in the mesh.
    z_min = min(b[0] for b in blocks)
    z_max = max(b[0] for b in blocks)
    n_z = z_max - z_min + 1
    x0 = min(b[2] for b in blocks)
    y0 = min(b[3] for b in blocks)
    x1 = max(b[2] + b[1].shape[1] for b in blocks)
    y1 = max(b[3] + b[1].shape[0] for b in blocks)
    shape = (n_z, y1 - y0, x1 - x0)
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
        vol[z - z_min, sy:sy + h, sx:sx + w] |= mask.astype(np.uint8)
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


from sam2_utils import video_viz

#: Largest canvas edge, in _sam px, before one chain is treated as an outlier. The
#: merged reprop render is the cautionary case: a single legacy chain whose window
#: covered the whole frame dragged every frame to full-frame size and the masks fell to
#: a fraction of a percent of the image.
CANVAS_CAP_PX = 900


def common_canvas(sizes, *, cap: int = CANVAS_CAP_PX):
    """``(h, w)`` big enough for every frame, capped so an outlier cannot size it."""
    if not sizes:
        raise ValueError("common_canvas needs at least one frame size")
    h = min(cap, max(s[0] for s in sizes))
    w = min(cap, max(s[1] for s in sizes))
    return int(h), int(w)


def fit_to_canvas(img: np.ndarray, canvas_hw, scale: float = 1.0) -> np.ndarray:
    """Scale ``img`` by ``scale``, then centre it on a ``canvas_hw`` black canvas.

    ``scale`` is how a chain at a different ``crop_scale`` is brought to the video's
    common nanometres per pixel. Padding without it would put two magnifications in one
    video with nothing on screen to say so. Anything still larger than the canvas after
    scaling is shrunk to fit, so a frame is never cropped silently.
    """
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h > ch or w > cw:
        f = min(ch / h, cw / w)
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    out = np.zeros((ch, cw, 3), dtype=np.uint8)
    oy, ox = (ch - h) // 2, (cw - w) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out


def _crop_scale(state: dict) -> int:
    cw = state.get("crop_window") or {}
    return int(cw.get("crop_scale") or 1)


def _chain_mask(chain_dir: Path, z) -> np.ndarray | None:
    """Read one chain's own saved mask for ``z``, or None if it has none.

    Deliberately NOT `pipeline.chain_masks_in_sam`: that remaps onto the shared
    scale-8 _sam grid for cross-chain aggregation (Task 4's volume), which is a
    different pixel size from the chain's own frames whenever crop_scale != 8. For
    display the chain's own mask PNG is already in the same space and the same pixel
    dimensions as the chain's own frames (verified on AIBL/chain_00: frame and mask
    both 1096x1208 at crop_scale 2), so no offset arithmetic is needed here at all.
    """
    if z is None:
        return None
    p = Path(chain_dir) / "masks" / f"mask_{int(z):04d}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 127


def neuron_video(root, neuron: str, out_path, *, fmt: str = "gif", progress=None):
    """One video for ``neuron``: every chain in z order, each in its own window, padded
    onto one canvas, captioned with chain and z so a problem is traceable back."""
    import json
    root = Path(root)
    kind = source_kind(root)
    chains = _chain_dirs(root, neuron)
    if not chains:
        print(f"[render] {neuron}: no chains, skipping video")
        return None

    loaded = []
    for ci, cdir in chains:
        mask_dir = cdir / "masks"
        if not mask_dir.is_dir() or not any(mask_dir.glob("mask_*.png")):
            continue
        state = json.loads((cdir / "state.json").read_text(encoding="utf-8"))
        frames = chain_frames(cdir, state, kind)
        f2z = {int(k): int(v) for k, v in (state.get("frame_to_z") or {}).items()}
        loaded.append((ci, cdir, state, frames, f2z))
    if not loaded:
        print(f"[render] {neuron}: no masks, skipping video")
        return None

    loaded.sort(key=lambda t: min(t[4].values()) if t[4] else 0)
    finest = min(_crop_scale(t[2]) for t in loaded)
    sizes = []
    for _ci, _cd, state, frames, _f2z in loaded:
        s = finest / _crop_scale(state)
        for img in frames.values():
            sizes.append((int(img.shape[0] * s), int(img.shape[1] * s)))
    canvas = common_canvas(sizes)

    tmp = Path("scratch_render") / f"review_{neuron}"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    segments, idx, total = {}, 0, sum(len(t[3]) for t in loaded)
    for ci, cdir, state, frames, f2z in loaded:
        s = finest / _crop_scale(state)
        for fi in sorted(frames):
            z = f2z.get(fi)
            img = fit_to_canvas(frames[fi], canvas, scale=s)
            _stamp(img, f"chain_{ci:02d}  z={z}")
            cv2.imwrite(str(tmp / f"{idx:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # Always give this frame an entry, even an empty one: to_gif/to_mp4 only
            # emit frames whose index is present in the dict, so a frame left out here
            # (rather than mapped to {}) would silently vanish from the video instead
            # of showing up as a chain with no mask at this z.
            segments[idx] = {}
            mask = _chain_mask(cdir, z)
            if mask is not None:
                seg = fit_to_canvas(np.stack([mask.astype(np.uint8) * 255] * 3, -1),
                                    canvas, scale=s)
                segments[idx] = {ci + 1: seg[..., 0] > 127}
            idx += 1
            if progress:
                progress(idx, total, f"{neuron} chain_{ci:02d}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = video_viz.to_gif if fmt == "gif" else video_viz.to_mp4
    writer(segments, tmp, out_path, obj_id=None, preview_scale=1, color=None)
    import shutil
    shutil.rmtree(tmp)
    print(f"[render] {neuron}: video {idx} frames -> {out_path}")
    return out_path


def _stamp(img_rgb: np.ndarray, text: str) -> None:
    """Burn a caption into the top-left, in place, dark then light so it stays readable
    over both bright cytoplasm and dark membrane."""
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img_rgb, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, thickness, cv2.LINE_AA)
