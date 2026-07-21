"""sam3_bakeoff.py: the 2x2 SAM2-vs-SAM3 bake-off harness (Task 6).

Runs FOUR cells on shared target-worm chains and scores them identically:

  sam2_prop      : SAM2 video predictor (setup.build_predictor(kind="video")),
                   driven by pipeline.propagate.propagate(...).
  sam2_perslice  : SAM2 image predictor (setup.build_predictor(kind="image")),
                   driven by pipeline.propagate.segment_per_slice(...).
  sam3_prop      : sam2_utils.sam3_backend.Sam3VideoPredictor, driven by the
                   SAME propagate(...) call as sam2_prop.
  sam3_perslice  : sam2_utils.sam3_backend.Sam3ImagePredictor, driven by the
                   SAME segment_per_slice(...) call as sam2_perslice.

Every cell shares ONE anchor-prompt build (predict.build_prompts + a single SAM2
image-mode anchor call, exactly the non-crop path pipeline.orchestrator.run_chain
takes) and ONE prepared _sam frame set (pipeline.prepare_video_frames), so the
2x2 differs only in which model ran, not in what it was seeded with. Reused,
not reinvented, from coprop_lab.py (load-a-chain pattern) and batch.py
(enumerate_chains / annotate_df / chains.json bootstrap).

Scoring reuses eval.merge_metric's own primitives (own_contained, foreign_hits,
MembraneSource) and sam2_utils.membrane's detectors (spanning_membrane,
underfill_fraction) directly against each cell's in-RAM video_segments, rather
than round-tripping through merge_metric.score_run's on-disk chain_masks_in_sam
reader: these masks never get saved via pipeline.save_masks (this is a driver
comparing predictors, not a chain run), so the disk-shaped API doesn't apply,
but the same definitions do:
  foreign_node_rate = fraction of scored frames containing >= 1 foreign node
  dropout           = fraction of scored frames empty or missing their own node
  underfill         = mean sam2_utils.membrane.underfill_fraction over frames
                       with an EM patch available
  mild_bleed        = fraction of EM-scored frames with a spanning membrane
                       ridge (sam2_utils.membrane.spanning_membrane) AND no
                       foreign-node hit (the node-only merge floor is Phase 0;
                       this is the finer Phase-2 signal)

Fail-fast + bounded-run rule (see .git/sdd/task-6-brief.md): the FIRST chain in
--chains runs its sam3_prop cell before anything else, as the fail-fast probe
for SAM3 reverse propagation (docs/explanation/sam3-bakeoff-findings.md, "Reverse
propagation verdict"). If that cell doesn't yield frames on both sides of the
anchor, the harness prints a clear message and stops before any further chain in
--chains (never silently limps on with a broken video adapter).

Usage
-----
    py -3 experiments/sam3_bakeoff.py --chains AIAL:5
    py -3 experiments/sam3_bakeoff.py                      # AIAL:5, then AIAL:0

Each cell is timed and VRAM-tracked independently (torch.cuda.max_memory_allocated,
reset per cell) and CUDA OOM is caught per cell and recorded as "Narval-only"
(this card is 6GB; a cell that doesn't fit is a cluster-only cell, not a harness
bug) so one OOM never aborts the run.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# repo-root bootstrap (this file lives in experiments/, batch.py/pipeline.py are
# root-level modules; same pattern as experiments/sam3_probe.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline
from pipeline import PipelineConfig
from pipeline.predict import centreline_by_z
from pipeline.propagate import segment_per_slice
from sam2_utils import config, membrane, setup
from sam2_utils.sam3_backend import (
    DEFAULT_CHECKPOINT_DIR,
    Sam3ImagePredictor,
    Sam3VideoPredictor,
)
from eval.merge_metric import (
    MembraneSource,
    foreign_hits,
    load_node_table,
    nodes_by_z,
    own_contained,
)
from batch import enumerate_chains

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard requirement to run cells,
    torch = None      # but the module should still import (e.g. for --help) without it.

CELL_NAMES = ["sam2_prop", "sam2_perslice", "sam3_prop", "sam3_perslice"]
DEFAULT_CHAINS = "AIAL:5,AIAL:0"          # short (17-frame) fail-fast chain, then the
                                          # long (113-node, anchor 56) chain
DEFAULT_OUT = Path("docs/figures/sam3-bakeoff")
FINDINGS_DOC = Path("docs/explanation/sam3-bakeoff-findings.md")


# =============================================================================
# GPU bookkeeping
# =============================================================================

def _free_gpu() -> None:
    """Best-effort GPU memory reclaim between cells (mirrors sam3_probe.py)."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: Exception) -> bool:
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# =============================================================================
# CLI
# =============================================================================

def parse_chains(spec: str) -> list[tuple[str, int]]:
    """'AIAL:5,AIAL:0' -> [("AIAL", 5), ("AIAL", 0)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        neuron, idx = part.split(":")
        out.append((neuron.strip(), int(idx)))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="2x2 SAM2-vs-SAM3 bake-off: {sam2,sam3} x {propagation,per-slice}.")
    ap.add_argument("--chains", default=DEFAULT_CHAINS,
                    help=f"comma-separated NEURON:chain_idx pairs (default {DEFAULT_CHAINS!r}); "
                         "the FIRST entry is treated as the fail-fast probe chain")
    ap.add_argument("--root", default=str(config.OUTPUT_ROOT),
                    help="output root, kept for parity with the other drivers (not written to "
                         "by this harness; only overlays under --out and the findings doc are)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_DIR, help="SAM3 HF checkpoint dir")
    ap.add_argument("--scale", type=int, default=8, help="SAM2/SAM3 _sam grid downscale")
    ap.add_argument("--model-size", default="large", choices=sorted(config.SAM2_CHECKPOINTS),
                    help="SAM2 checkpoint size (SAM3 has one size)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="overlay PNG output dir")
    return ap


# =============================================================================
# Chain inputs: built ONCE per chain, shared byte-identical across all 4 cells
# =============================================================================

@dataclass
class ChainInputs:
    neuron: str
    chain_idx: int
    frames_dir: str
    frame_to_z: dict[int, int]
    anchor_frame_idx: int
    n_frames: int
    prompts: "pipeline.Prompts"       # box_sam + points_sam/labels, _sam space
    centreline_tif: dict[int, tuple[float, float]]
    obj_id: int = 1


def build_chain_inputs(neuron: str, chain_idx: int, chain: dict, annotate_df: pd.DataFrame,
                       cfg: PipelineConfig, image_predictor) -> ChainInputs:
    """One anchor pick + one prompt build + one prepared _sam frame set for this chain.

    Mirrors pipeline.orchestrator.run_chain's plain (non-tier-2) path exactly: select_anchor,
    build_prompts, one image-mode anchor predict (to size the video-seed box), then
    prepare_video_frames. `image_predictor` is a throwaway SAM2 image predictor the caller
    builds and frees around this call; the same box+points Prompts (and the same frames_dir)
    are then handed unchanged to all four cells, so the 2x2 measures the model, not the seed.
    """
    anchor_node_id, anchor_catmaid_z = pipeline.select_anchor(chain, annotate_df)
    image_sam, _full_hw = pipeline.load_frame_sam(anchor_catmaid_z, scale=cfg.scale, frame_store=None)
    prompts = pipeline.build_prompts(anchor_node_id, anchor_catmaid_z, annotate_df,
                                     scale=cfg.scale, k_max_neg=cfg.k_max_neg,
                                     neg_radius=cfg.neg_radius)
    mask_anchor, image_score, _logits = pipeline.image_predict(
        image_predictor, image_sam, prompts,
        multimask=cfg.multimask_anchor,
        select_contain_radius_px=cfg.qc_skeleton_dilation_px,
        select_area_bounds=(cfg.gate_min_area_frac, cfg.gate_max_area_frac))
    box = pipeline.box_from_mask(mask_anchor, margin=cfg.box_margin, image_hw_sam=mask_anchor.shape[:2])
    if box is None:
        raise RuntimeError(
            f"{neuron}/chain_{chain_idx:02d}: empty anchor mask (image_score={image_score:.3f}); "
            "cannot seed the bake-off for this chain")
    prompts.box_sam = np.asarray(box, dtype=np.float32)
    image_predictor.reset_predictor()

    frames_dir, frame_to_z, anchor_frame_idx, n_frames = pipeline.prepare_video_frames(
        chain, annotate_df, scale=cfg.scale, frames_root=cfg.frames_root,
        anchor_catmaid_z=anchor_catmaid_z, neuron=neuron, chain_idx=chain_idx, frame_store=None)
    centreline_tif = centreline_by_z(chain, annotate_df)
    print(f"[bakeoff] {neuron}/chain_{chain_idx:02d}: anchor node {anchor_node_id} "
          f"(z={anchor_catmaid_z}, frame_idx={anchor_frame_idx}), {n_frames} frames, "
          f"anchor image_score={image_score:.3f}")
    return ChainInputs(neuron=neuron, chain_idx=chain_idx, frames_dir=frames_dir,
                       frame_to_z=frame_to_z, anchor_frame_idx=anchor_frame_idx,
                       n_frames=n_frames, prompts=prompts, centreline_tif=centreline_tif)


# =============================================================================
# The four cells
# =============================================================================

def run_sam2_prop(inputs: ChainInputs, cfg: PipelineConfig):
    vp, _dev = setup.build_predictor(size=cfg.model_size, kind="video", image_size=cfg.image_size)
    try:
        video_segments, _conf, _iou = pipeline.propagate(
            vp, inputs.frames_dir, inputs.prompts, inputs.anchor_frame_idx, obj_id=inputs.obj_id)
        return video_segments
    finally:
        del vp


def run_sam2_perslice(inputs: ChainInputs, cfg: PipelineConfig, annotate_df: pd.DataFrame):
    ip, _dev = setup.build_predictor(size=cfg.model_size, kind="image", image_size=cfg.image_size)
    try:
        video_segments, _conf, _iou = segment_per_slice(
            ip, inputs.frames_dir, inputs.frame_to_z, inputs.centreline_tif,
            annotate_df, cfg=cfg, obj_id=inputs.obj_id, cw=None)
        return video_segments
    finally:
        del ip


def run_sam3_prop(inputs: ChainInputs, checkpoint_dir: str):
    vp = Sam3VideoPredictor(checkpoint_dir)
    try:
        video_segments, _conf, _iou = pipeline.propagate(
            vp, inputs.frames_dir, inputs.prompts, inputs.anchor_frame_idx, obj_id=inputs.obj_id)
        return video_segments
    finally:
        del vp


def run_sam3_perslice(inputs: ChainInputs, cfg: PipelineConfig, annotate_df: pd.DataFrame,
                      checkpoint_dir: str):
    ip = Sam3ImagePredictor(checkpoint_dir)
    try:
        video_segments, _conf, _iou = segment_per_slice(
            ip, inputs.frames_dir, inputs.frame_to_z, inputs.centreline_tif,
            annotate_df, cfg=cfg, obj_id=inputs.obj_id, cw=None)
        return video_segments
    finally:
        del ip


def _dispatch(name: str, inputs: ChainInputs, cfg: PipelineConfig, annotate_df: pd.DataFrame,
             checkpoint_dir: str):
    if name == "sam2_prop":
        return run_sam2_prop(inputs, cfg)
    if name == "sam2_perslice":
        return run_sam2_perslice(inputs, cfg, annotate_df)
    if name == "sam3_prop":
        return run_sam3_prop(inputs, checkpoint_dir)
    if name == "sam3_perslice":
        return run_sam3_perslice(inputs, cfg, annotate_df, checkpoint_dir)
    raise ValueError(f"unknown cell {name!r}")


# =============================================================================
# Per-cell run wrapper: timing + VRAM + OOM containment
# =============================================================================

@dataclass
class CellResult:
    name: str
    neuron: str
    chain_idx: int
    video_segments: Optional[dict] = None
    seconds: float = float("nan")
    peak_vram_gb: float = float("nan")
    status: str = "ok"                 # "ok" | "Narval-only" | "error"
    error: Optional[str] = None


def _run_cell(name: str, neuron: str, chain_idx: int, fn) -> CellResult:
    _free_gpu()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        video_segments = fn()
        seconds = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9
               if (torch is not None and torch.cuda.is_available()) else float("nan"))
        return CellResult(name, neuron, chain_idx, video_segments, seconds, peak, "ok")
    except Exception as e:                       # noqa: BLE001 - one bad cell must not kill the run
        seconds = time.perf_counter() - t0
        if _is_oom(e):
            print(f"[bakeoff] {name} on {neuron}/chain_{chain_idx:02d}: CUDA OOM "
                  "-> recording as Narval-only, continuing")
            status = "Narval-only"
        else:
            print(f"[bakeoff] {name} on {neuron}/chain_{chain_idx:02d}: FAILED "
                  f"({type(e).__name__}: {e})")
            traceback.print_exc()
            status = "error"
        return CellResult(name, neuron, chain_idx, None, seconds, float("nan"), status, str(e))
    finally:
        _free_gpu()


def _reverse_ok(res: CellResult, inputs: ChainInputs) -> bool:
    """True if `res` (a sam3_prop/sam2_prop CellResult) carries frames on both sides of
    the anchor where the chain has frames to carry (the fail-fast reverse-propagation
    check; see docs/explanation/sam3-bakeoff-findings.md)."""
    if res.status != "ok" or not res.video_segments:
        return False
    idxs = set(res.video_segments)
    ok = True
    if inputs.anchor_frame_idx > 0:
        ok = ok and any(i < inputs.anchor_frame_idx for i in idxs)
    if inputs.anchor_frame_idx < inputs.n_frames - 1:
        ok = ok and any(i > inputs.anchor_frame_idx for i in idxs)
    return ok


# =============================================================================
# Scoring: eval.merge_metric's own primitives + sam2_utils.membrane, applied
# directly to a cell's in-RAM video_segments (see module docstring).
# =============================================================================

def score_cell(video_segments: dict[int, dict[int, np.ndarray]], frame_to_z: dict[int, int],
              neuron: str, nbz: dict[int, list[tuple[float, float, str, str]]],
              membrane_source: MembraneSource, *, obj_id: int = 1,
              radius: int = 3, tau: float = membrane.DEFAULT_TAU) -> dict:
    """foreign_node_rate / dropout / underfill / mild_bleed over one cell's frames.

    Masks are already full-frame _sam (no chain-crop offset in this harness), so the
    node grid `nbz` (nodes_by_z's _sam-scaled coordinates) applies with x0=y0=0,
    exactly like eval.merge_metric.score_chain's per-frame use of the same helpers.
    """
    n_frames = n_foreign_frames = n_dropout = n_mem_scored = n_mild_bleed = 0
    underfill_vals: list[float] = []
    for frame_idx, seg in sorted(video_segments.items()):
        if obj_id not in seg:
            continue
        z = frame_to_z.get(frame_idx)
        if z is None:
            continue
        mask = np.asarray(seg[obj_id])
        mask = (mask[0] if mask.ndim == 3 else mask).astype(bool)
        n_frames += 1

        nodes = nbz.get(int(z), [])
        own_xy = [(x, y) for (x, y, cell, _nid) in nodes if cell == neuron]
        own_ok = any(own_contained(mask, 0, 0, xy, radius) for xy in own_xy) if own_xy else False
        if (not mask.any()) or (not own_ok):
            n_dropout += 1
        fids = foreign_hits(mask, 0, 0, nodes, neuron, radius)
        if fids:
            n_foreign_frames += 1

        h, w = mask.shape
        mem = membrane_source.map_for(int(z), 0, 0, h, w)
        if mem is not None:
            spanning, _frac = membrane.spanning_membrane(mask, mem, tau=tau)
            underfill_vals.append(membrane.underfill_fraction(mask, mem, tau=tau))
            n_mem_scored += 1
            if spanning and not fids:
                n_mild_bleed += 1

    return {
        "n_frames": n_frames,
        "foreign_node_rate": (n_foreign_frames / n_frames) if n_frames else float("nan"),
        "dropout": (n_dropout / n_frames) if n_frames else float("nan"),
        "underfill": (float(np.mean(underfill_vals)) if underfill_vals else float("nan")),
        "mild_bleed": (n_mild_bleed / n_mem_scored) if n_mem_scored else float("nan"),
    }


def _row_for(res: CellResult, inputs: ChainInputs, nbz, membrane_source: MembraneSource) -> dict:
    row = {"cell": res.name, "neuron": res.neuron, "chain_idx": res.chain_idx,
          "status": res.status, "seconds": res.seconds, "peak_vram_gb": res.peak_vram_gb,
          "foreign_node_rate": float("nan"), "dropout": float("nan"),
          "underfill": float("nan"), "mild_bleed": float("nan"), "n_frames": 0}
    if res.status == "ok" and res.video_segments:
        row.update(score_cell(res.video_segments, inputs.frame_to_z, inputs.neuron,
                              nbz, membrane_source, obj_id=inputs.obj_id))
    return row


# =============================================================================
# Overlays + tables
# =============================================================================

def save_overlay(res: CellResult, inputs: ChainInputs, out_dir: Path) -> Optional[Path]:
    if res.status != "ok" or not res.video_segments:
        return None
    from sam2_utils import video_viz as vv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = vv.grid(res.video_segments, inputs.frames_dir, obj_id=inputs.obj_id,
                 frame_to_z=inputs.frame_to_z, anchor_idx=inputs.anchor_frame_idx,
                 n=min(12, inputs.n_frames), preview_scale=4, alpha=0.5, cols=4)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inputs.neuron}_chain{inputs.chain_idx:02d}_{res.name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[bakeoff] wrote overlay -> {out_path}")
    return out_path


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return f"{v:.{nd}f}"


def format_table(rows: list[dict]) -> str:
    header = (f"{'cell':<14}{'status':<13}{'foreign_node_rate':>18}{'dropout':>10}"
             f"{'underfill':>11}{'mild_bleed':>12}{'seconds':>10}{'peak_vram_gb':>14}")
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['cell']:<14}{r['status']:<13}{_fmt(r['foreign_node_rate']):>18}"
            f"{_fmt(r['dropout']):>10}{_fmt(r['underfill']):>11}{_fmt(r['mild_bleed']):>12}"
            f"{_fmt(r['seconds'], 2):>10}{_fmt(r['peak_vram_gb'], 3):>14}")
    return "\n".join(lines)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| chain | cell | status | foreign_node_rate | dropout | underfill | "
        "mild_bleed | seconds | peak_vram_gb |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['neuron']}/chain_{r['chain_idx']:02d} | {r['cell']} | {r['status']} | "
            f"{_fmt(r['foreign_node_rate'])} | {_fmt(r['dropout'])} | {_fmt(r['underfill'])} | "
            f"{_fmt(r['mild_bleed'])} | {_fmt(r['seconds'], 2)} | {_fmt(r['peak_vram_gb'], 3)} |")
    return "\n".join(lines)


def append_findings_doc(rows: list[dict], path: Path = FINDINGS_DOC) -> None:
    """Fill in the doc's placeholder "Bake-off results" section with the real table."""
    if not rows or not path.exists():
        return
    text = path.read_text()
    placeholder = (
        "To be filled in by the harness (Task 6): a four-row table over\n"
        "{SAM2, SAM3} x {propagation, per-slice} on AIAL chain_05 and chain_00.\n"
    )
    table_md = markdown_table(rows)
    if placeholder in text:
        text = text.replace(placeholder, table_md + "\n")
    else:
        text = text.rstrip("\n") + "\n\n" + table_md + "\n"
    path.write_text(text)
    print(f"[bakeoff] appended results table -> {path}")


# =============================================================================
# Driver
# =============================================================================

def run_bakeoff(args: argparse.Namespace) -> list[dict]:
    cfg = PipelineConfig(model_size=args.model_size, scale=args.scale, save_downscale=args.scale,
                         k_max_neg=3, neg_radius=150, box_margin=10,
                         output_root=Path(args.root), frames_root=config.FRAMES_ROOT)
    annotate_df = load_node_table()
    with open(config.CHAINS_PATH) as f:
        chains = json.load(f)
    lookup = {(n, i): c for n, i, c in enumerate_chains(chains, neurons=None)}

    nbz = nodes_by_z(annotate_df, cfg.scale)
    membrane_source = MembraneSource(cfg.scale)

    chain_specs = parse_chains(args.chains)
    out_dir = Path(args.out)

    all_rows: list[dict] = []
    for ci, (neuron, chain_idx) in enumerate(chain_specs):
        print(f"\n{'=' * 70}\n{neuron}/chain_{chain_idx:02d}\n{'=' * 70}")
        key = (neuron, chain_idx)
        if key not in lookup:
            print(f"[bakeoff] {neuron}/chain_{chain_idx:02d} not found in "
                  f"{config.CHAINS_PATH}; skipping")
            continue

        anchor_ip, _dev = setup.build_predictor(size=cfg.model_size, kind="image",
                                                image_size=cfg.image_size)
        try:
            inputs = build_chain_inputs(neuron, chain_idx, lookup[key], annotate_df, cfg, anchor_ip)
        except Exception as e:
            print(f"[bakeoff] {neuron}/chain_{chain_idx:02d}: chain build FAILED "
                  f"({type(e).__name__}: {e})")
            traceback.print_exc()
            del anchor_ip
            _free_gpu()
            continue
        del anchor_ip
        _free_gpu()

        # Fail-fast rule: the FIRST chain in --chains runs sam3_prop before anything
        # else (the reverse-propagation probe); every other chain runs in the fixed
        # CELL_NAMES order.
        cell_order = (["sam3_prop"] + [n for n in CELL_NAMES if n != "sam3_prop"]
                     if ci == 0 else list(CELL_NAMES))

        results: dict[str, CellResult] = {}
        fail_fast = False
        for name in cell_order:
            print(f"\n[bakeoff] --- {name} ---")
            res = _run_cell(name, neuron, chain_idx,
                            lambda name=name: _dispatch(name, inputs, cfg, annotate_df,
                                                        args.checkpoint))
            results[name] = res
            if name == "sam3_prop":
                ok = _reverse_ok(res, inputs)
                print(f"[bakeoff] fail-fast probe (sam3_prop reverse propagation): "
                     f"{'OK' if ok else 'FAILED'} (status={res.status})")
                if not ok and ci == 0:
                    fail_fast = True

        chain_rows: list[dict] = []
        for name in CELL_NAMES:
            res = results.get(name)
            if res is None:
                continue
            row = _row_for(res, inputs, nbz, membrane_source)
            chain_rows.append(row)
            save_overlay(res, inputs, out_dir)
        all_rows.extend(chain_rows)
        print("\n" + format_table(chain_rows))

        if fail_fast:
            print(f"\n[bakeoff] FAIL-FAST: sam3_prop did not propagate bidirectionally from "
                 f"the anchor on {neuron}/chain_{chain_idx:02d} (the probe chain). Stopping "
                 "before any further --chains entries per the bounded-run rule.")
            break

    if all_rows:
        append_findings_doc(all_rows)
    print("\n[bakeoff] done.")
    return all_rows


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_bakeoff(args)


if __name__ == "__main__":
    main()
