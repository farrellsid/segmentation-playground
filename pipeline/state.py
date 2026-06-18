"""Chain structs (Prompts, ChainState, AnchorScore) and ChainState <-> state.json (de)serialization.

Three things plain json won't handle on its own:
  - numpy arrays in Prompts (points / labels / box) -> lists on dump, arrays on load
  - frame_to_z keys come back as strings           -> cast to int on load
  - Path fields in PipelineConfig                   -> str on dump, Path on load
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PipelineConfig


@dataclass
class Prompts:
    """SAM2-space prompts for one chain's anchor frame."""
    points_sam: np.ndarray            # (N, 2) float, _sam space
    labels: np.ndarray                # (N,) int, 1 = positive / 0 = negative
    box_sam: Optional[np.ndarray] = None   # (4,) xyxy float, _sam space; None until box_from_mask


@dataclass
class ChainState:
    """
    Everything needed to run, pause, resume, or re-open one chain.

    Persist this to <neuron>/chain_<idx>/state.json. It holds *references and
    metadata*, never the mask arrays themselves -> those live on disk under
    masks/. video_segments stays in RAM during a run and is reconstructed from
    PNGs if you re-open the chain.
    """
    neuron: str                       # = the notebook's TARGET_CELL_NAME (identity, not a knob)
    chain_idx: int
    status: str = "pending"           # pending / running / done / flagged / failed

    # anchor (filled by select_anchor)
    anchor_node_id: Optional[int] = None
    anchor_catmaid_z: Optional[int] = None
    anchor_frame_idx: Optional[int] = None     # filled once video frames are prepped

    # prompts (filled by build_prompts, updated by box_from_mask / GUI edits)
    prompts: Optional[Prompts] = None

    # image-phase result summary (mask itself goes to disk)
    image_score: Optional[float] = None

    # anchor-quality gate verdict, as a plain JSON-ready dict ->
    # parallel to qc_summary. Filled by score_anchor in run_chain. Logs this as
    # the per-chain "anchor verdict" feature for the learned P(error) detector.
    anchor_score: Optional[dict] = None

    # video input metadata (filled by prepare_video_frames)
    frames_dir: Optional[str] = None
    frame_to_z: Optional[dict[int, int]] = None
    n_frames: Optional[int] = None

    # tier-2 per-chain crop window (alignment.CropWindow.to_dict()), or None for the
    # _sam full-frame path. When set, this chain's masks/frames/prompts all live in
    # `_pcrop` (the crop space) rather than _sam, and QC/review/GUI rebuild the
    # CropWindow from this to map skeleton nodes + clicks. Filled by run_chain when
    # cfg.chain_crop is on.
    crop_window: Optional[dict] = None

    # tier-2 SAFETY: True when cfg.chain_crop was requested but the
    # per-chain crop anchor was poor, so this chain was re-run in the plain _sam path
    # (crop_window is then None and masks/frames are _sam, exactly like a full-frame run).
    # Recorded for how often the fallback fires, and as a P(error) feature.
    fell_back_to_sam: bool = False

    # tier-2 fallback DIAGNOSTICS (captured from the CROP pass before the _sam recovery
    # pass overwrites image_score/anchor_score, otherwise the failing crop-pass values
    # are lost and the final state.json only shows the healthy _sam recovery, making it
    # impossible to tell WHY a chain fell back). None unless fell_back_to_sam is True.
    #   fellback_reason   : which trigger fired, "empty-mask" / "gate(...)" / "score<0.7"
    #   crop_image_score  : the crop-pass anchor image_score (the over-zoom tell)
    #   crop_anchor_score : the crop-pass anchor gate verdict (score_anchor dict)
    fellback_reason: Optional[str] = None
    crop_image_score: Optional[float] = None
    crop_anchor_score: Optional[dict] = None

    # qc summary + triage
    qc_summary: Optional[dict] = None          # flag counts, worst frames, etc.
    triage_frames: list[int] = field(default_factory=list)

    obj_id: int = 1                            # per-chain; increments for multi-obj merge

    # snapshot of the run settings this chain was processed under (reproducibility):
    # a resumed/re-opened chain replays with the knobs it actually ran under, even
    # if the global defaults have since drifted.
    config: PipelineConfig = field(default_factory=PipelineConfig)

    # runtime telemetry, filled by run_chain's per-phase timer (_step/_finish).
    # Declared as real fields (not stamped-on attributes) so they serialise with
    # the rest of the state: batch.py reads them right after a run to write
    # _timing.csv, and persisting them keeps a resumed/re-opened chain's timing.
    phase_seconds: dict = field(default_factory=dict)        # {phase label: seconds}
    phase_subseconds: dict = field(default_factory=dict)     # {sub-step label: seconds}


@dataclass
class AnchorScore:
    """Threshold-light quality verdict for one chain's anchor (image-phase) mask.

    The geometry here is judged entirely in _sam space -> the space image_predict
    works in, and the space prompts.points_sam already lives in -> so there is *no*
    coordinate transform in this function (deliberately: the anchor mask and the
    positive prompt point share one frame). That keeps it off the bug-prone
    transform path.

    Three sub-checks, mirroring the gate:
      contained        -> does the mask cover the positive (skeleton) prompt point,
                          within a small radius? Tri-state, same meaning as
                          qc.skeleton_contained but encoded JSON-clean:
                          True / False / None(no positive point -> abstain).
      n_components,
      largest_cc_frac   -> single-CC health: fraction of foreground in the largest
                          connected component (a clean anchor is ~one blob).
      area_frac         -> foreground as a fraction of the frame: floored to catch an
                          empty/near-empty mask, ceiled to catch a runaway grab of
                          background.

    `passed` is the AND of the enabled checks; an abstaining (None) containment does
    not fail. `reasons` lists the checks that fired, reusing the qc vocabulary
    ('noskel' / 'area' / 'frag') so the gate and the per-frame QC speak the same
    language downstream.
    """
    contained: Optional[bool]
    n_components: int
    largest_cc_frac: float
    area_frac: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _anchor_score_to_dict(s: AnchorScore) -> dict:
    """JSON-ready plain dict (no numpy types) for ChainState.anchor_score."""
    return {
        "contained": None if s.contained is None else bool(s.contained),
        "n_components": int(s.n_components),
        "largest_cc_frac": float(s.largest_cc_frac),
        "area_frac": float(s.area_frac),
        "passed": bool(s.passed),
        "reasons": list(s.reasons),
    }


# =============================================================================
# Serialization  (ChainState <-> state.json)
# =============================================================================

def _prompts_to_dict(p: Optional[Prompts]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "points_sam": np.asarray(p.points_sam).tolist(),
        "labels": np.asarray(p.labels).tolist(),
        "box_sam": None if p.box_sam is None else np.asarray(p.box_sam).tolist(),
    }


def _prompts_from_dict(d: Optional[dict]) -> Optional[Prompts]:
    if d is None:
        return None
    box = d.get("box_sam")
    return Prompts(
        points_sam=np.array(d["points_sam"], dtype=float),
        labels=np.array(d["labels"], dtype=int),
        box_sam=None if box is None else np.array(box, dtype=np.float32),
    )


def _config_to_dict(c: PipelineConfig) -> dict:
    d = asdict(c)
    d["output_root"] = None if c.output_root is None else str(c.output_root)
    d["frames_root"] = None if c.frames_root is None else str(c.frames_root)
    return d


def _config_from_dict(d: Optional[dict]) -> PipelineConfig:
    d = dict(d or {})
    if d.get("output_root") is not None:
        d["output_root"] = Path(d["output_root"])
    if d.get("frames_root") is not None:
        d["frames_root"] = Path(d["frames_root"])
    return PipelineConfig(**d)


def state_to_dict(state: ChainState) -> dict:
    """Plain-json-safe dict view of a ChainState."""
    ftz = state.frame_to_z
    return {
        "neuron": state.neuron,
        "chain_idx": state.chain_idx,
        "status": state.status,
        "anchor_node_id": state.anchor_node_id,
        "anchor_catmaid_z": state.anchor_catmaid_z,
        "anchor_frame_idx": state.anchor_frame_idx,
        "prompts": _prompts_to_dict(state.prompts),
        "image_score": None if state.image_score is None else float(state.image_score),
        "anchor_score": state.anchor_score,
        "frames_dir": state.frames_dir,
        "frame_to_z": None if ftz is None else {str(k): int(v) for k, v in ftz.items()},
        "n_frames": state.n_frames,
        "crop_window": state.crop_window,
        "fell_back_to_sam": bool(state.fell_back_to_sam),
        "fellback_reason": state.fellback_reason,
        "crop_image_score": (None if state.crop_image_score is None
                             else float(state.crop_image_score)),
        "crop_anchor_score": state.crop_anchor_score,
        "qc_summary": state.qc_summary,
        "triage_frames": list(state.triage_frames),
        "obj_id": state.obj_id,
        "config": _config_to_dict(state.config),
        "phase_seconds": dict(state.phase_seconds),
        "phase_subseconds": dict(state.phase_subseconds),
    }


def state_from_dict(d: dict) -> ChainState:
    ftz = d.get("frame_to_z")
    return ChainState(
        neuron=d["neuron"],
        chain_idx=d["chain_idx"],
        status=d.get("status", "pending"),
        anchor_node_id=d.get("anchor_node_id"),
        anchor_catmaid_z=d.get("anchor_catmaid_z"),
        anchor_frame_idx=d.get("anchor_frame_idx"),
        prompts=_prompts_from_dict(d.get("prompts")),
        image_score=d.get("image_score"),
        anchor_score=d.get("anchor_score"),
        frames_dir=d.get("frames_dir"),
        frame_to_z=None if ftz is None else {int(k): int(v) for k, v in ftz.items()},
        n_frames=d.get("n_frames"),
        crop_window=d.get("crop_window"),
        fell_back_to_sam=bool(d.get("fell_back_to_sam", False)),
        fellback_reason=d.get("fellback_reason"),
        crop_image_score=d.get("crop_image_score"),
        crop_anchor_score=d.get("crop_anchor_score"),
        qc_summary=d.get("qc_summary"),
        triage_frames=list(d.get("triage_frames", [])),
        obj_id=d.get("obj_id", 1),
        config=_config_from_dict(d.get("config")),
        phase_seconds=dict(d.get("phase_seconds", {}) or {}),
        phase_subseconds=dict(d.get("phase_subseconds", {}) or {}),
    )


def save_state(state: ChainState, path: str | Path) -> Path:
    """Serialize a ChainState to state.json (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_to_dict(state), indent=2))
    return path


def load_state(path: str | Path) -> ChainState:
    """Reload a ChainState from state.json."""
    return state_from_dict(json.loads(Path(path).read_text()))
