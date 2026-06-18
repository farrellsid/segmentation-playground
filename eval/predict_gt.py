"""
predict_gt.py, run the SAM2 pipeline on SEM-Dauer 1's EM.

DISCONTINUED (kept for reference only). This was the points-only scaffold that validated the
Stage-0 eval harness end to end and surfaced the registration problem. The real production
pipeline now runs on SEM-Dauer 1 directly (Stage 0.2: `batch.py --preset eval` via the
`pipeline.FrameStore` seam + `eval.gt_dataset`, scored by `eval.score_batch`), so this module
is no longer the scored path and is not maintained. Its unfinished bleed levers (cross-neuron
negatives, box seed, postprocess) are retired with it. The code below is left intact for
historical reference.

Goal: produce predictions for the cross-worm GT (SEM-Dauer 1, project 280) so
the eval ruler can measure the *current* pipeline. Output feeds both
scorers with no further glue:

    GT_PRED_DIR/
      masks/<neuron>/<slice:03d>.png   binary per-neuron masks  -> eval.score.DirPredictionSource
      labelmaps/pred_s###.png          uint16 per-slice labelmaps -> eval.run_erl --mode pred

How this differs from pipeline.py (why it's a separate module)
--------------------------------------------------------------
`pipeline.py` runs in the *target* worm's frame: EM comes from the TIF stack
(`config.WORM_PATH`) indexed by `catmaid_z`, prompts are built in that worm's
`_sam` space. SEM-Dauer 1 is a *different* stack:
  * EM is PNG slices in `config.GT_EM_DIR` indexed by the VAST slice z (1:1 with
    the p280 skeleton z), read here via `GroundTruth.em_slice`.
  * Prompts come from the p280 skeletons, mapped into the GT pixel grid with the
    fitted `eval.registration.Registration` (skel stack-px -> GT mask/EM px).
  * Output lands directly on the GT grid (same H×W as the GT labelmaps), so
    score.py / run_erl can consume it without resampling.

Pipeline (all wired)
--------------------
  - load_inputs()            : chains + skeleton table + registration + GroundTruth
  - chain_prompt_points()    : skeleton nodes of a chain -> {z: [(x,y) in GT px]}
  - build_predictors()       : SAM2 image + video predictors (setup.build_predictor)
  - predict_chain()          : transcode EM -> cached JPEG view, seed the densest
                               slice with positive points, propagate bidirectionally
                               (pipeline.propagate), return {z: GT-grid bool mask}
  - write_neuron_masks()     : per-neuron binary PNGs (score.py layout)
  - composite_labelmaps()    : per-neuron masks -> per-slice uint16 labelmaps
  - run() orchestration + CLI

The seed is positive-points-only; the obvious accuracy levers (image-mode anchor
refinement, cross-neuron negatives, box seed) are noted in predict_chain.

pipeline/torch are imported lazily inside build_predictors/predict_chain so a bare
`import eval.predict_gt` stays cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam2_utils import config, setup, diagnostics
from sam2_utils.skeletons import normalize_name
from .erl import _canon_id
from .groundtruth import GroundTruth
from .registration import Registration


# =============================================================================
# Config
# =============================================================================

@dataclass
class PredictGTConfig:
    """Inputs/outputs + knobs for a GT-worm prediction run."""
    skeleton_csv: Path = field(
        default_factory=lambda: config.DATA_DIR / "groundtruth" / "skeletons_p280" / "aggregate_data_pv.csv")
    chains_path: Path = field(
        default_factory=lambda: config.DATA_DIR / "groundtruth" / "skeletons_p280" / "chains.json")
    registration_path: Path = field(
        default_factory=lambda: config.DATA_DIR / "groundtruth" / "skeletons_p280" / "registration.json")
    pred_dir: Path = field(default_factory=lambda: Path(config.GT_PRED_DIR))
    model_size: str = config.DEFAULT_MODEL_SIZE
    device: str = "cuda"
    # which neurons to run (None = all in chains); handy for a small first pass
    neuron_limit: Optional[int] = None

    @property
    def masks_dir(self) -> Path:
        return self.pred_dir / "masks"

    @property
    def labelmaps_dir(self) -> Path:
        return self.pred_dir / "labelmaps"


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class PredictInputs:
    chains: List[dict]
    registration: Registration
    gt: GroundTruth
    node_xyz: Dict[str, Tuple[float, float, float]]   # canonical node id -> stack-px
    grid_hw: Tuple[int, int]                            # GT mask grid (H, W)


def load_inputs(cfg: PredictGTConfig) -> PredictInputs:
    """Load chains, registration, GroundTruth, and a node->xyz lookup.

    Node positions come straight from the skeleton CSV (stack-px); ids are
    canonicalized to match chain node ids (see erl._canon_id for the float-vs-int
    parent_id gotcha).
    """
    import pandas as pd
    chains = json.loads(Path(cfg.chains_path).read_text())
    reg = Registration.from_json(cfg.registration_path)
    gt = GroundTruth.from_config()

    df = pd.read_csv(cfg.skeleton_csv, dtype={"node_id": str}).dropna(subset=["x", "y", "z"])
    node_xyz = {_canon_id(nid): (float(x), float(y), float(z))
                for nid, x, y, z in zip(df["node_id"], df["x"], df["y"], df["z"])}

    # GT grid dims from any present slice (all slices share H×W)
    z0 = gt.slice_indices[0]
    H, W = gt.label_slice(z0).shape[:2]

    if cfg.neuron_limit:
        keep = sorted({normalize_name(c["cell_name"]) for c in chains})[:cfg.neuron_limit]
        keep = set(keep)
        chains = [c for c in chains if normalize_name(c["cell_name"]) in keep]

    return PredictInputs(chains=chains, registration=reg, gt=gt,
                         node_xyz=node_xyz, grid_hw=(int(H), int(W)))


def chain_prompt_points(chain: dict, inp: PredictInputs) -> Dict[int, List[Tuple[float, float]]]:
    """Map a chain's skeleton nodes into the GT pixel grid, grouped by slice z.

    Returns ``{z: [(x_px, y_px), ...]}``, candidate positive point prompts for
    SAM2 on each GT EM slice. z is the VAST slice index (1:1 with skeleton z).
    Real nodes are used (virtual nodes too, since they sit on intermediate slices
    and give a prompt on every section the chain crosses).
    """
    reg = inp.registration
    by_z: Dict[int, List[Tuple[float, float]]] = {}
    for raw in chain.get("nodes", []):
        nid = _canon_id(raw)
        p = inp.node_xyz.get(nid)
        if p is None:
            continue
        x, y, z = p
        zi = int(round(z))
        xy = reg.transform(np.array([[x, y]], dtype=float), zi)[0]
        by_z.setdefault(zi, []).append((float(xy[0]), float(xy[1])))
    return by_z


# =============================================================================
# SAM2 core
# =============================================================================

def build_predictors(cfg: PredictGTConfig):
    """Load SAM2 image + video predictors.

    Built the same way run_aval.py / pipeline.py construct them
    (`sam2.build_sam2`, `SAM2ImagePredictor`, `build_sam2_video_predictor`) using
    `config.SAM2_CHECKPOINTS[cfg.model_size]`. Returns whatever predict_chain needs.
    """

    image_pred, _ = setup.build_predictor(size=cfg.model_size, kind="image")
    video_pred, _ = setup.build_predictor(size=cfg.model_size, kind="video")
    
    diagnostics.snapshot("after model loading")
    
    # Return predictors as list: 0 is image, 1 is video
    return [image_pred, video_pred]


def _ensure_em_cache(cfg: PredictGTConfig, gt: GroundTruth, z_list: Sequence[int]) -> Path:
    """Transcode each GT EM slice to a JPEG ONCE, under ``pred_dir/frames_cache``.

    SAM2's video predictor reads JPEGs off disk; the GT EM is PNG. Decoding every
    chain's z-range from the HDD on every chain (9.7k chains, heavy overlap) would
    thrash the disk, so we cache the transcode keyed by z and reuse it across chains
    (mirrors pipeline._ensure_cached_frames, minus the downscale, EM is already 1/4).
    """
    from PIL import Image
    cache = Path(cfg.pred_dir) / "frames_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for z in z_list:
        p = cache / f"z{int(z):04d}.jpg"
        if p.exists():
            continue
        em = gt.em_slice(z)
        if em.ndim == 2:                                  # grayscale -> 3-channel for SAM2
            em = np.stack([em, em, em], axis=-1)
        Image.fromarray(em.astype(np.uint8)).save(p, quality=95)
    return cache


def _build_frame_view(cfg: PredictGTConfig, cache: Path,
                      z_range: Sequence[int]) -> Tuple[Path, Dict[int, int]]:
    """Make a contiguous 0-indexed JPEG view SAM2's init_state wants.

    Hard-links the cached frames (same volume as pred_dir, so links are free and
    cross-volume errors can't happen; falls back to copy). Returns (view_dir,
    frame_idx->z). Caller deletes the view dir when done; the cache persists.
    """
    import os
    import shutil
    import tempfile
    views_root = Path(cfg.pred_dir) / "_views"
    views_root.mkdir(parents=True, exist_ok=True)
    view = Path(tempfile.mkdtemp(dir=views_root))
    frame_to_z: Dict[int, int] = {}
    for i, z in enumerate(z_range):
        src = cache / f"z{int(z):04d}.jpg"
        dst = view / f"{i:05d}.jpg"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)
        frame_to_z[i] = int(z)
    return view, frame_to_z


def predict_chain(
    cfg: PredictGTConfig,
    chain: dict,
    prompt_points: Dict[int, List[Tuple[float, float]]],
    inp: PredictInputs,
    predictors,
) -> Dict[int, np.ndarray]:
    """Run SAM2 for one chain over the GT EM frames -> ``{z: bool mask on GT grid}``.

    EM access is already solved (``inp.gt.em_slice(z)``, the F: ``one_fourth_scale``
    PNGs are GT-grid already, no TIF decode/downscale). The rest reuses pipeline.py's
    frame-agnostic ``propagate`` unchanged:
      1. Contiguous frame view over the chain's z-span (cached JPEG transcode of EM).
      2. Anchor = the slice carrying the most skeleton points (densest seed); its
         points (already in GT px via the registration) become positive prompts.
      3. ``pipeline.propagate`` seeds the anchor + tracks bidirectionally; the returned
         masks are at frame resolution == the GT grid, so no resampling.

    Seeds POSITIVE POINTS only (no image-mode anchor refinement, no cross-neuron
    negatives, no box). Those are the obvious next levers if masks bleed into
    neighbours (see pipeline.image_predict / build_prompts' negative construction).
    """
    import shutil
    import pipeline
    from pipeline import Prompts

    video_pred = predictors[1]            # build_predictors returns [image, video]
    gt = inp.gt

    # in-range slices that actually have prompts + GT frames, contiguous for SAM2
    zs = [z for z in sorted(prompt_points) if gt.has_slice(z)]
    if not zs:
        return {}
    z_range = [z for z in range(zs[0], zs[-1] + 1) if gt.has_slice(z)]
    if not z_range:
        return {}

    cache = _ensure_em_cache(cfg, gt, z_range)
    view, frame_to_z = _build_frame_view(cfg, cache, z_range)
    z_to_frame = {z: i for i, z in frame_to_z.items()}

    # anchor = MIDDLE slice of the chain's prompted z-range, so propagation runs
    # both ways (mirrors pipeline.select_anchor's mid node). NB do NOT pick the
    # "densest" slice: decomposed chains have ~1 node/slice, so all slices tie and
    # max() returns the first -> anchor pinned to frame 0 -> reverse pass yields 0
    # frames and SAM2 tracks one-way from a chain tip (its worst-defined end).
    anchor_z = zs[len(zs) // 2]
    anchor_idx = z_to_frame[anchor_z]
    pts = np.asarray(prompt_points[anchor_z], dtype=float)      # (N,2) GT px == frame px
    prompts = Prompts(points_sam=pts, labels=np.ones(len(pts), dtype=int))

    try:
        video_segments, _conf, _piou = pipeline.propagate(
            video_pred, str(view), prompts, anchor_idx,
            obj_id=1, seed_box=False, seed_points=True)
    finally:
        shutil.rmtree(view, ignore_errors=True)             # keep the cache, drop the view

    out: Dict[int, np.ndarray] = {}
    for fi, per_obj in video_segments.items():
        z = frame_to_z.get(int(fi))
        m = per_obj.get(1)
        if z is None or m is None:
            continue
        m = np.asarray(m, dtype=bool)
        if m.ndim == 3:               # SAM2 yields (1, H, W) per object, drop the channel
            m = m[0]
        if m.any():
            out[z] = m
    return out


# =============================================================================
# Output writers
# =============================================================================

def write_neuron_masks(pred_dir: Path, neuron: str, masks_by_z: Dict[int, np.ndarray]) -> int:
    """Write binary per-neuron masks as ``masks/<neuron>/<z:03d>.png``.

    Unions onto any existing mask for the slice (a neuron split across several
    chains accumulates). Returns the number of slices written. This is exactly the
    layout `eval.score.DirPredictionSource` reads.
    """
    from PIL import Image
    out = Path(pred_dir) / "masks" / neuron
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for z, mask in masks_by_z.items():
        p = out / f"{int(z):03d}.png"
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3:                                   # tolerate a (1, H, W) mask
            m = m[0]
        if p.exists():
            m = m | (np.asarray(Image.open(p)) > 0)
        Image.fromarray((m.astype(np.uint8) * 255)).save(p)
        n += 1
    return n


def composite_labelmaps(
    cfg: PredictGTConfig,
    grid_hw: Tuple[int, int],
    slice_indices: Sequence[int],
    *,
    neuron_ids: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Combine per-neuron mask PNGs into per-slice uint16 labelmaps.

    Reads ``masks/<neuron>/<z:03d>.png`` and writes ``labelmaps/pred_s###.png``
    where pixel value == that neuron's object id (0 == background). Returns the
    ``{neuron: id}`` map (also saved as ``labelmaps/neuron_ids.json``).

    Overlap policy (FIRST WRITER WINS) is a deliberate placeholder: when two
    neurons' predicted masks overlap, the lower-id neuron keeps the pixel and the
    collision is counted. TODO: decide the real policy, overlaps are exactly the
    *merge* signal ERL cares about, so you may instead want to mark contested
    pixels (e.g. a reserved MERGE id) or keep per-neuron labelmaps and resolve in
    the metric. Logged so the simplification is never silent.
    """
    from PIL import Image
    masks_dir = Path(cfg.masks_dir)
    lm_dir = Path(cfg.labelmaps_dir)
    lm_dir.mkdir(parents=True, exist_ok=True)
    H, W = grid_hw

    neurons = sorted(p.name for p in masks_dir.iterdir() if p.is_dir()) if masks_dir.exists() else []
    if neuron_ids is None:
        neuron_ids = {neu: i + 1 for i, neu in enumerate(neurons)}   # 1..N, 0=bg
    (lm_dir / "neuron_ids.json").write_text(json.dumps(neuron_ids, indent=2))

    total_collisions = 0
    for z in slice_indices:
        lab = np.zeros((H, W), dtype=np.uint16)
        written = np.zeros((H, W), dtype=bool)
        for neu in neurons:
            p = masks_dir / neu / f"{int(z):03d}.png"
            if not p.exists():
                continue
            m = np.asarray(Image.open(p)) > 0
            collide = m & written
            total_collisions += int(collide.sum())
            take = m & ~written                       # first writer wins
            lab[take] = neuron_ids[neu]
            written |= m
        # uint16 array -> Pillow infers mode I;16 (don't pass mode=, deprecated)
        Image.fromarray(lab).save(lm_dir / f"pred_s{int(z):03d}.png")
    if total_collisions:
        print(f"[predict_gt] WARNING: {total_collisions} overlap px resolved first-writer-wins "
              "(see composite_labelmaps docstring, overlaps are the merge signal).")
    return neuron_ids


# =============================================================================
# Orchestration + CLI
# =============================================================================

def run(cfg: PredictGTConfig, *, clean: bool = False) -> None:
    """End-to-end: predict every chain, write masks, composite labelmaps.

    ``clean=True`` wipes ``masks/`` + ``labelmaps/`` first (but keeps
    ``frames_cache/``, so the EM transcode is reused). Use it on every re-run:
    ``write_neuron_masks`` UNIONS onto existing PNGs, so without a clean a re-run
    would OR new masks into the previous run's, silently contaminating them.
    """
    if clean:
        import shutil
        for d in (cfg.masks_dir, cfg.labelmaps_dir):
            shutil.rmtree(d, ignore_errors=True)
        print(f"[predict_gt] clean: wiped masks/ + labelmaps/ under {cfg.pred_dir} "
              "(kept frames_cache/)")

    inp = load_inputs(cfg)
    predictors = build_predictors(cfg)

    written_z: set = set()
    for chain in inp.chains:
        neuron = normalize_name(chain["cell_name"])
        pts = chain_prompt_points(chain, inp)
        if not pts:
            continue
        masks_by_z = predict_chain(cfg, chain, pts, inp, predictors)
        write_neuron_masks(cfg.pred_dir, neuron, masks_by_z)
        written_z.update(masks_by_z)

    composite_labelmaps(cfg, inp.grid_hw, sorted(written_z))
    print(f"[predict_gt] done -> {cfg.pred_dir}")


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run SAM2 on SEM-Dauer 1 (predictions for eval).")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--neuron-limit", type=int, default=None)
    ap.add_argument("--model-size", default=config.DEFAULT_MODEL_SIZE)
    ap.add_argument("--clean", action="store_true",
                    help="wipe masks/ + labelmaps/ before running (keeps frames_cache/); "
                         "use on every re-run since mask writes union onto existing files")
    args = ap.parse_args()
    cfg = PredictGTConfig(model_size=args.model_size, neuron_limit=args.neuron_limit)
    if args.pred_dir:
        cfg.pred_dir = args.pred_dir
    run(cfg, clean=args.clean)


if __name__ == "__main__":
    _main()
