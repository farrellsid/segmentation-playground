"""
video_viz.py — watch SAM2 video-mode propagation results.

Drop this in your `sam2_utils/` package (or next to the notebook) and import it.
It overlays the per-frame masks in `video_segments` onto the JPEG frames that
SAM2 actually saw (the ones written to `video_frames_dir`).

Why read frames from `video_frames_dir` instead of the tifs?
  Those JPEGs are 0-indexed and already at SCALE resolution, so frame `i` lines
  up exactly with `video_segments[i]` and the SCALE-space masks. No tif re-reads,
  no resampling, no coordinate math — what you see is what SAM2 saw.

Main entry points:
  animate(...)    -> inline scrubber/player (returns IPython HTML), best for *watching*
  grid(...)       -> static figure, N evenly-spaced frames, good for a quick glance / commit
  to_mp4(...)     -> write an .mp4 to disk (needs ffmpeg)
  to_gif(...)     -> write a .gif to disk (pillow, no ffmpeg needed)
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import to_rgb

# Stable per-object palette (obj_id -> RGB float). Cycles for >10 objects.
_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3]


def _color_for(obj_id: int):
    return _PALETTE[int(obj_id) % len(_PALETTE)]


def _load_frame(frames_dir: Path, idx: int, preview_scale: int = 1) -> np.ndarray:
    """Read frame `idx` from the SAM2 JPEG folder as RGB uint8."""
    p = Path(frames_dir) / f"{idx:05d}.jpg"
    if not p.exists():
        raise FileNotFoundError(f"frame {idx} not found at {p}")
    img = cv2.imread(str(p))                       # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if preview_scale and preview_scale != 1:
        img = cv2.resize(
            img, None, fx=1 / preview_scale, fy=1 / preview_scale,
            interpolation=cv2.INTER_AREA,
        )
    return img


def _overlay(frame_rgb: np.ndarray, mask_bool: np.ndarray, color, alpha: float):
    """Alpha-blend a single binary mask onto a frame. mask resized to frame if needed."""
    h, w = frame_rgb.shape[:2]
    if mask_bool.shape[:2] != (h, w):
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    out = frame_rgb.astype(np.float32)
    col = np.asarray(color, dtype=np.float32) * 255.0
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * col
    return out.astype(np.uint8), mask_bool


def _objs_at(video_segments, idx):
    """{obj_id: bool mask} for a frame, squeezing any leading channel dim."""
    seg = video_segments.get(idx, {})
    out = {}
    for oid, m in seg.items():
        m = np.asarray(m)
        if m.ndim == 3:          # SAM2 often returns (1, H, W)
            m = m[0]
        out[oid] = m.astype(bool)
    return out


def _frame_indices(video_segments, frames_dir):
    """Propagated frame indices that also have a JPEG on disk, sorted."""
    have = {int(p.stem) for p in Path(frames_dir).glob("*.jpg")}
    idxs = sorted(i for i in video_segments if i in have)
    return idxs


def _label(idx, frame_to_z, area, anchor_idx, total):
    z = frame_to_z.get(idx) if frame_to_z else None
    bits = [f"frame {idx}/{total - 1}"]
    if z is not None:
        bits.append(f"z={z}")
    bits.append(f"area={area:,}px")
    if anchor_idx is not None and idx == anchor_idx:
        bits.append("ANCHOR")
    return "   ".join(bits)


# --------------------------------------------------------------------------- #
# Inline player — the one you want for "let me actually see it"
# --------------------------------------------------------------------------- #
def animate(
    video_segments,
    frames_dir,
    obj_id=None,
    frame_to_z=None,
    anchor_idx=None,
    preview_scale=2,
    alpha=0.5,
    fps=12,
    max_frames=None,
    figsize=(7, 7),
):
    """
    Build an inline scrubber/player of the propagated masks.

    obj_id      : int, list of ints, or None (None = all objects present).
    preview_scale: extra downsample *on top of* SCALE, just for display size.
                   Bumps this up if the notebook file gets huge (each frame is
                   embedded as a PNG). 2 is a good default at SCALE=8.
    max_frames  : cap the number of frames embedded (evenly subsampled). Useful
                  for long ~340-frame chains; None = all.

    Returns an IPython.display.HTML — just `return` it as the last line of a cell,
    or do `from IPython.display import display; display(animate(...))`.
    """
    from IPython.display import HTML

    idxs = _frame_indices(video_segments, frames_dir)
    if not idxs:
        raise ValueError(
            "No frames to show: video_segments keys don't match JPEGs in "
            f"{frames_dir}. Did you run the propagation + frame-prep cells?"
        )

    total = len(idxs)
    if max_frames and total > max_frames:
        sel = np.linspace(0, total - 1, max_frames).round().astype(int)
        idxs = [idxs[i] for i in sel]

    if isinstance(obj_id, int):
        want = {obj_id}
    elif obj_id is None:
        want = None
    else:
        want = set(obj_id)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)  # don't double-render in notebooks
    ax.axis("off")

    first = _load_frame(frames_dir, idxs[0], preview_scale)
    im = ax.imshow(first)
    title = ax.set_title("")

    def render(i):
        idx = idxs[i]
        frame = _load_frame(frames_dir, idx, preview_scale)
        objs = _objs_at(video_segments, idx)
        if want is not None:
            objs = {o: m for o, m in objs.items() if o in want}
        area = 0
        for oid, m in objs.items():
            frame, m_rs = _overlay(frame, m, _color_for(oid), alpha)
            area += int(m_rs.sum())
        im.set_data(frame)
        title.set_text(_label(idx, frame_to_z, area, anchor_idx, len(idxs)))
        return im, title

    anim = animation.FuncAnimation(
        fig, render, frames=len(idxs), interval=1000 / fps, blit=False
    )
    return HTML(anim.to_jshtml(fps=fps))


# --------------------------------------------------------------------------- #
# Static grid — quick glance, survives notebook reloads, good for sharing
# --------------------------------------------------------------------------- #
def grid(
    video_segments,
    frames_dir,
    obj_id=None,
    frame_to_z=None,
    anchor_idx=None,
    n=12,
    preview_scale=4,
    alpha=0.5,
    cols=4,
):
    """N evenly-spaced overlaid frames in a grid. Returns the matplotlib Figure."""
    idxs = _frame_indices(video_segments, frames_dir)
    if not idxs:
        raise ValueError(
            f"No frames to show: video_segments keys don't match JPEGs in {frames_dir}."
        )

    # Always include the anchor frame if it's in range.
    sel = np.linspace(0, len(idxs) - 1, min(n, len(idxs))).round().astype(int)
    chosen = [idxs[i] for i in sel]
    if anchor_idx is not None and anchor_idx in idxs and anchor_idx not in chosen:
        chosen[len(chosen) // 2] = anchor_idx
        chosen = sorted(set(chosen))

    if isinstance(obj_id, int):
        want = {obj_id}
    elif obj_id is None:
        want = None
    else:
        want = set(obj_id)

    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, chosen):
        frame = _load_frame(frames_dir, idx, preview_scale)
        objs = _objs_at(video_segments, idx)
        if want is not None:
            objs = {o: m for o, m in objs.items() if o in want}
        area = 0
        for oid, m in objs.items():
            frame, m_rs = _overlay(frame, m, _color_for(oid), alpha)
            area += int(m_rs.sum())
        ax.imshow(frame)
        is_anchor = anchor_idx is not None and idx == anchor_idx
        ax.set_title(
            _label(idx, frame_to_z, area, anchor_idx, len(idxs)),
            fontsize=8,
            color="crimson" if is_anchor else "black",
            fontweight="bold" if is_anchor else "normal",
        )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Disk exports
# --------------------------------------------------------------------------- #
def _frames_iter(video_segments, frames_dir, obj_id, preview_scale, alpha):
    idxs = _frame_indices(video_segments, frames_dir)
    if isinstance(obj_id, int):
        want = {obj_id}
    elif obj_id is None:
        want = None
    else:
        want = set(obj_id)
    for idx in idxs:
        frame = _load_frame(frames_dir, idx, preview_scale)
        objs = _objs_at(video_segments, idx)
        if want is not None:
            objs = {o: m for o, m in objs.items() if o in want}
        for oid, m in objs.items():
            frame, _ = _overlay(frame, m, _color_for(oid), alpha)
        yield idx, frame


def to_gif(video_segments, frames_dir, out_path, obj_id=None,
           preview_scale=4, alpha=0.5, fps=12):
    """Write an animated GIF (uses pillow; no ffmpeg needed)."""
    from PIL import Image
    frames = [Image.fromarray(f) for _, f in
              _frames_iter(video_segments, frames_dir, obj_id, preview_scale, alpha)]
    if not frames:
        raise ValueError("nothing to write")
    out_path = str(out_path)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, disposal=2)
    return out_path


def to_mp4(video_segments, frames_dir, out_path, obj_id=None,
           preview_scale=2, alpha=0.5, fps=12):
    """Write an .mp4 (uses cv2's VideoWriter; mp4v codec)."""
    items = list(_frames_iter(video_segments, frames_dir, obj_id, preview_scale, alpha))
    if not items:
        raise ValueError("nothing to write")
    h, w = items[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(out_path)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for _, frame in items:
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    return out_path
