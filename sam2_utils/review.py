"""
review.py — read-only proofreading viewer for SAM2 mask stacks *on disk*.

A sibling of ``video_viz``. Where ``video_viz`` overlays the in-RAM
``video_segments`` dict a propagation run just produced, this module rebuilds the
same overlay from the **saved** artifacts of a finished chain::

    output_root/<neuron>/chain_NN/
        state.json                  # frame_to_z, frames_dir, anchor_frame_idx,
                                    #   triage_frames, obj_id
        masks/mask_<catmaid_z>.png  # 0/255 uint8, written by pipeline.save_masks
        qc.csv                      # optional (written in M2): flag columns if present

so you can re-open and proofread a chain long after the run, without re-running
SAM2 or keeping anything in memory.

Read-only by design
-------------------
This is the proofreading tool, NOT the M4 intervention GUI — no point editing, no
re-prompting (that is napari, and per PIPELINE_CONTEXT §4 there is exactly one
correction tool). Adding click-to-edit here is precisely the thing to *not* do; it
would grow this into the second GUI that doc warns against.

Single source of truth
-----------------------
It deliberately reuses two helpers so it can't drift from the rest of the package:
  - frame JPEGs + overlay/animate/grid rendering -> ``sam2_utils.video_viz``
  - how a mask PNG is read off disk              -> ``sam2_utils.qc``
    (``_iter_mask_paths`` parses ``mask_<catmaid_z>.png``; ``_load_binary``
    thresholds ``> 0``). Using these underscore helpers across the package is
    intentional: "how a mask is read" must have one definition.

Coordinate spaces
-----------------
Masks are stored at _sam (the canonical rule ``save_downscale == scale``), and the
JPEG frames are downscaled by the same ``scale``, so a mask and its frame share
resolution and no coordinate math is needed — the same invariant ``video_viz``
relies on. (If you ever set ``save_downscale != scale``, ``video_viz._overlay``
nearest-resizes the mask to the frame, so it still displays, just softer.)

A note on filenames: only ``mask_<catmaid_z>.png`` (no ``z`` prefix) is parsed,
matching ``pipeline.save_masks`` and ``qc._iter_mask_paths``. The old notebook's
``mask_z<...>.png`` files are skipped (same as qc) — re-run through the pipeline.

Entry points mirror video_viz
------------------------------
    load_chain(chain_dir)            -> ReviewData (segments rebuilt from disk)
    animate(chain_dir, ...)          -> inline scrubber/player (notebook)
    grid(chain_dir, ...)             -> static N-frame grid (works anywhere)
    animate_flagged / grid_flagged   -> same, but only the QC-flagged frames
    to_mp4(chain_dir, out) / to_gif  -> write a proof to disk

All accept either a chain directory (Path/str) or a pre-loaded ``ReviewData``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from . import video_viz as vv
from . import qc as _qc


PathLike = Union[str, Path]

### EXAMPLE USAGE

# from sam2_utils import review
# chain_dir = OUTPUT_ROOT / "AVAL" / "chain_02"

# review.animate(chain_dir)          # scrubber in a notebook
# review.grid(chain_dir, n=16)       # static glance, works in a script too
# review.grid_flagged(chain_dir)     # only QC-flagged frames (once qc.csv exists)
# review.to_gif(chain_dir, "aval_chain02.gif")


@dataclass
class ReviewData:
    """Everything the viewers need, rebuilt from a chain's on-disk artifacts."""
    video_segments: dict[int, dict[int, np.ndarray]]    # {frame_idx: {obj_id: bool mask}}
    frames_dir: str
    frame_to_z: dict[int, int]                          # {frame_idx: catmaid_z}
    anchor_idx: Optional[int] = None
    triage_frames: list[int] = field(default_factory=list)  # frame_idx of flagged frames
    obj_id: int = 1
    qc: object = None                                   # pandas DataFrame if qc.csv present
    title: str = ""

    @property
    def n_frames(self) -> int:
        return len(self.video_segments)


# ---------------------------------------------------------------------------
# Disk -> ReviewData
# ---------------------------------------------------------------------------

def _read_state(chain_dir: Path) -> dict:
    """Parse state.json, casting frame_to_z keys back to int."""
    sp = chain_dir / "state.json"
    if not sp.exists():
        raise FileNotFoundError(
            f"no state.json in {chain_dir}; pass frames_dir= and frame_to_z= "
            f"explicitly to review a chain without one."
        )
    d = json.loads(sp.read_text())
    ftz = d.get("frame_to_z")
    d["frame_to_z"] = (None if ftz is None
                       else {int(k): int(v) for k, v in ftz.items()})
    return d


def load_chain(
    chain_dir: PathLike,
    *,
    masks_dir: Optional[PathLike] = None,
    frames_dir: Optional[PathLike] = None,
    frame_to_z: Optional[dict[int, int]] = None,
    obj_id: Optional[int] = None,
    anchor_idx: Optional[int] = None,
    with_qc: bool = True,
    triage_is_z: bool = True,
    warn_unmapped: bool = True,
    verbose: bool = True,
) -> ReviewData:
    """Rebuild a chain's overlay inputs from its saved artifacts.

    Parameters
    ----------
    chain_dir : path
        ``.../<neuron>/chain_NN``. Expected to contain ``state.json`` and
        ``masks/``. Both can be overridden by the kwargs below.
    masks_dir : path, optional
        Defaults to ``chain_dir/masks``.
    frames_dir, frame_to_z, obj_id, anchor_idx : optional
        Normally read from ``state.json``; pass to override or to review a chain
        that has no state.json (then frames_dir and frame_to_z are required).
    with_qc : bool
        If True and ``chain_dir/qc.csv`` exists, load it and derive the flagged
        frame list from its ``flag`` column.
    triage_is_z : bool
        How to interpret ``state.json``'s ``triage_frames`` when qc.csv is absent.
        Defaults to True: entries are CATMAID-z values (consistent with qc, the
        mask filenames, and ``qc.show_flagged``, which are all z-keyed). Set False
        if a future milestone decides to store raw frame_idx there instead. qc.csv,
        when present, is always z-keyed and takes priority regardless.

    Returns
    -------
    ReviewData
    """
    chain_dir = Path(chain_dir)

    # state.json supplies the frame<->z map, frames_dir, anchor, obj_id, triage —
    # any of which the caller may override. Only read it if something's missing.
    state: dict = {}
    if (frames_dir is None or frame_to_z is None
            or anchor_idx is None or obj_id is None or with_qc):
        try:
            state = _read_state(chain_dir)
        except FileNotFoundError:
            if frames_dir is None or frame_to_z is None:
                raise
            state = {}

    frames_dir = str(frames_dir if frames_dir is not None else state["frames_dir"])
    frame_to_z = frame_to_z if frame_to_z is not None else state.get("frame_to_z")
    if frame_to_z is None:
        raise ValueError(
            "frame_to_z is required but is absent from state.json and was not passed."
        )
    anchor_idx = anchor_idx if anchor_idx is not None else state.get("anchor_frame_idx")
    obj_id = obj_id if obj_id is not None else int(state.get("obj_id", 1))

    masks_dir = Path(masks_dir) if masks_dir is not None else chain_dir / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"no masks dir at {masks_dir}")

    # Rebuild {frame_idx: {obj_id: bool mask}} from disk, keyed to match the JPEGs.
    z_to_frame = {z: idx for idx, z in frame_to_z.items()}
    segments: dict[int, dict[int, np.ndarray]] = {}
    unmapped = 0
    for z, path in _qc._iter_mask_paths(masks_dir):      # mask_<catmaid_z>.png
        frame_idx = z_to_frame.get(int(z))
        if frame_idx is None:
            unmapped += 1
            continue
        segments[frame_idx] = {obj_id: _qc._load_binary(path)}   # >0 -> bool
    if not segments:
        raise ValueError(
            f"no usable masks in {masks_dir} (matched against {len(frame_to_z)} "
            f"frames). Are these mask_<catmaid_z>.png files from this chain?"
        )
    if warn_unmapped and unmapped:
        print(f"[review] {unmapped} mask file(s) had a z not in frame_to_z; skipped")

    # Optional QC table (written in M2). Derive flagged frames for the *_flagged views.
    qc_df = None
    triage_z: list[int] = []
    if with_qc:
        qc_csv = chain_dir / "qc.csv"
        if qc_csv.exists():
            import pandas as pd
            qc_df = pd.read_csv(qc_csv)
            if "z" in qc_df.columns:
                qc_df = qc_df.set_index("z")
            _col = next((c for c in ("queue", "flag") if c in qc_df.columns), None)
            if _col is not None:
                triage_z = [int(z) for z in qc_df.index[qc_df[_col].astype(bool)]]
    if not triage_z and state.get("triage_frames"):
        raw = [int(v) for v in state["triage_frames"]]
        triage_z = raw if triage_is_z else [
            frame_to_z[i] for i in raw if i in frame_to_z
        ]
    triage_frames = [z_to_frame[z] for z in triage_z if z in z_to_frame]

    title = f"{state.get('neuron', '?')} chain {state.get('chain_idx', '?')}"
    if verbose:
        anc = "" if anchor_idx is None else f", anchor frame_idx={anchor_idx}"
        print(f"[review] {title}: {len(segments)} frames{anc}, "
              f"{len(triage_frames)} flagged")

    return ReviewData(
        video_segments=segments,
        frames_dir=frames_dir,
        frame_to_z=frame_to_z,
        anchor_idx=anchor_idx,
        triage_frames=triage_frames,
        obj_id=obj_id,
        qc=qc_df,
        title=title,
    )


# ---------------------------------------------------------------------------
# Internal: accept a path or a pre-loaded ReviewData
# ---------------------------------------------------------------------------

def _as_data(chain: Union[PathLike, ReviewData]) -> ReviewData:
    if isinstance(chain, ReviewData):
        return chain
    return load_chain(chain)


def _subset(data: ReviewData, frame_idxs: Sequence[int]) -> ReviewData:
    """Return a copy whose video_segments is restricted to ``frame_idxs``."""
    keep = set(frame_idxs)
    seg = {i: m for i, m in data.video_segments.items() if i in keep}
    return replace(data, video_segments=seg)


# ---------------------------------------------------------------------------
# Viewers — thin wrappers that delegate rendering to video_viz
# ---------------------------------------------------------------------------

def animate(chain: Union[PathLike, ReviewData], *, obj_id: Optional[int] = None,
            preview_scale: int = 2, alpha: float = 0.5, fps: int = 12,
            max_frames: Optional[int] = None, figsize=(7, 7)):
    """Inline scrubber/player of the saved masks. Returns IPython HTML (notebook).

    For disk locations other than the defaults, ``load_chain(..., masks_dir=...)``
    first and pass the resulting ReviewData here.
    """
    data = _as_data(chain)
    return vv.animate(
        data.video_segments, data.frames_dir,
        obj_id=obj_id if obj_id is not None else data.obj_id,
        frame_to_z=data.frame_to_z, anchor_idx=data.anchor_idx,
        preview_scale=preview_scale, alpha=alpha, fps=fps,
        max_frames=max_frames, figsize=figsize,
    )


def grid(chain: Union[PathLike, ReviewData], *, obj_id: Optional[int] = None,
         n: int = 12, preview_scale: int = 4, alpha: float = 0.5, cols: int = 4):
    """N evenly-spaced overlaid frames in a static grid. Returns a Figure."""
    data = _as_data(chain)
    return vv.grid(
        data.video_segments, data.frames_dir,
        obj_id=obj_id if obj_id is not None else data.obj_id,
        frame_to_z=data.frame_to_z, anchor_idx=data.anchor_idx,
        n=n, preview_scale=preview_scale, alpha=alpha, cols=cols,
    )


def _no_flags_fig(data: ReviewData):
    import matplotlib.pyplot as plt
    print(f"[review] {data.title}: no flagged frames "
          f"(no qc.csv / triage info, or nothing flagged)")
    return plt.figure()


def animate_flagged(chain: Union[PathLike, ReviewData], **kwargs):
    """Like ``animate`` but restricted to QC-flagged frames."""
    data = _as_data(chain)
    if not data.triage_frames:
        return _no_flags_fig(data)
    return animate(_subset(data, data.triage_frames), **kwargs)


def grid_flagged(chain: Union[PathLike, ReviewData], **kwargs):
    """Like ``grid`` but restricted to QC-flagged frames — the M2 threshold-eyeball view.

    Note: for *why* each frame flagged (the signal breakdown + EM context +
    skeleton marker), ``qc.show_flagged`` is the richer tool. This is the quick
    "are these flags sane" glance over the same masks.
    """
    data = _as_data(chain)
    if not data.triage_frames:
        return _no_flags_fig(data)
    return grid(_subset(data, data.triage_frames), **kwargs)


# ---------------------------------------------------------------------------
# Disk exports — share a proof without opening a notebook
# ---------------------------------------------------------------------------

def to_mp4(chain: Union[PathLike, ReviewData], out_path: PathLike, *,
           obj_id: Optional[int] = None, preview_scale: int = 2,
           alpha: float = 0.5, fps: int = 12) -> str:
    """Write an .mp4 of the overlaid masks (needs ffmpeg via cv2)."""
    data = _as_data(chain)
    return vv.to_mp4(
        data.video_segments, data.frames_dir, out_path,
        obj_id=obj_id if obj_id is not None else data.obj_id,
        preview_scale=preview_scale, alpha=alpha, fps=fps,
    )


def to_gif(chain: Union[PathLike, ReviewData], out_path: PathLike, *,
           obj_id: Optional[int] = None, preview_scale: int = 4,
           alpha: float = 0.5, fps: int = 12) -> str:
    """Write an animated GIF of the overlaid masks (pillow; no ffmpeg needed)."""
    data = _as_data(chain)
    return vv.to_gif(
        data.video_segments, data.frames_dir, out_path,
        obj_id=obj_id if obj_id is not None else data.obj_id,
        preview_scale=preview_scale, alpha=alpha, fps=fps,
    )
