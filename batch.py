"""
batch.py: headless batch runner + resume.

This is run_aval.py generalized into a loop. Same session setup (predictors,
annotate_df, chains built once), then run *every* chain unattended, recording
status to a manifest ledger as it goes, and rolling the per-chain QC flags up
into one cross-chain triage queue.

What this is and isn't
----------------------
IS:  run all chains overnight, survive crashes (resume from the manifest),
     never recompute a finished chain, and produce `_triage.csv` so you can
     measure the auto-flag rate across the whole dataset before building the GUI.
ISN'T: mid-propagation halt-and-re-prompt. That's the `propagate` generator
     restructure, coupled to the napari GUI, and lives in the GUI. This runner treats
     each chain as a single atomic `run_chain` call: run it, record what QC
     flagged, move on. Resist wiring interventions in here.

Storage
-------
    output/
      _manifest.csv                 # every chain x status: drives batch + resume
      _triage.csv                   # flagged frames across all chains: feeds the GUI
      <neuron>/chain_<idx:02d>/
        state.json                  # ChainState (save_state/load_state)
        qc.csv                      # per-frame metrics (indexed by catmaid_z)
        masks/mask_<catmaid_z:04d>.png

Usage
-----
Run configs are named **presets** in `sam2_utils/presets.py` (which worm, paths, model,
tier-2/gif, default neurons); pick one with `--preset` and override any field with a flag:

    python batch.py                              # = --preset original (target worm defaults)
    py -3 batch.py --preset original --neurons AVAL AVAR --clean
    py -3 batch.py --preset eval --neurons URYVL    # SEM-Dauer 1 cross-worm GT eval
    py -3 batch.py --preset eval --neuron-limit 3   # first N neurons; --all = every neuron (guarded)
    # then score a GT run:  py -3 -m eval.score_batch --preset eval
    # or import run_batch / build_triage_queue from a notebook for inspection.

The GT path runs the SAME pipeline on a different worm via a `pipeline.FrameStore` (EM
source) + registration-baked prompts: see eval/gt_dataset.py + eval/README.md.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

# Project imports. `pipeline` is the library at the repo root; the
# sam2_utils pieces are the stable helpers.
import pipeline
from pipeline import ChainState, PipelineConfig, save_state  # load_state if resuming state
from sam2_utils import setup, alignment, diagnostics, review, config

# DATA PATHS: defined once in sam2_utils.config; edit them there, not here.
CSV_PATH    = config.CSV_PATH
CHAINS_PATH = config.CHAINS_PATH
OUTPUT_ROOT = config.OUTPUT_ROOT
FRAMES_ROOT = config.FRAMES_ROOT     # SAM2 JPEG frame folders go here

# Run configurations (which worm, paths, PipelineConfig knobs, tier-2/gif, default
# neurons) live in sam2_utils/presets.py: `--preset eval|original`. Edit presets there.
from sam2_utils import presets

# Status vocabulary (matches ChainState.status).
PENDING, RUNNING, DONE, FLAGGED, FAILED = (
    "pending", "running", "done", "flagged", "failed",
)
COMPLETE_STATUSES = {DONE, FLAGGED}      # ran to completion; don't re-run on resume

MANIFEST_COLUMNS = [
    "neuron", "chain_idx", "status",
    "n_frames", "n_flagged", "n_intervene", "flag_rate",
    "anchor_frame_idx",
    # anchor-quality gate verdict, rolled up per chain so it sits
    # next to the QC summary and joins to _triage.csv on (neuron, chain_idx).
    # anchor_reasons is the comma-joined fail list ('' = passed); anchor_contained
    # is True/False/'' (blank = abstained, no positive point).
    "anchor_passed", "anchor_reasons", "anchor_contained",
    "anchor_lcc", "anchor_area_frac",
    "error", "updated_at",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write to a temp file in the same dir, then rename. A crash mid-write
    can't corrupt the manifest: you either get the old file or the new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _t = perf_counter()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)
    rec = _IO_STATS.setdefault(path.name, [0.0, 0])
    rec[0] += perf_counter() - _t
    rec[1] += 1


# =============================================================================
# Chain enumeration
# =============================================================================

def enumerate_chains(
    chains: Sequence[dict],
    neurons: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, dict]]:
    """Flatten chains.json into [(neuron, chain_idx, chain_dict), ...].

    chain_idx is the position *within that neuron's* chain list, the same index
    the notebook uses (`cell_chain.index(subchain)`), so it matches the on-disk
    `chain_<idx:02d>` folders and ChainState.chain_idx.

    `neurons`: optional allow-list of cell names (e.g. ["AVAL", "AVAR"]) to scope
    a partial batch run; None = every neuron.
    """
    by_neuron: "OrderedDict[str, list]" = OrderedDict()
    for ch in chains:
        by_neuron.setdefault(ch["cell_name"], []).append(ch)

    out: List[Tuple[str, int, dict]] = []
    for neuron, chs in by_neuron.items():
        if neurons is not None and neuron not in neurons:
            continue
        for idx, ch in enumerate(chs):
            out.append((neuron, idx, ch))
    return out

# =============================================================================
# Runtime telemetry
# =============================================================================
_PHASES = ["select anchor", "load anchor frame", "build prompts",
           "image-mode prediction", "box from mask", "prepare video frames",
           "propagate (bidirectional)", "save masks", "qc + flag"]

def _append_timing(output_root, neuron, chain_idx, state, peak_vram):
    ps  = getattr(state, "phase_seconds", {}) or {}
    sub = getattr(state, "phase_subseconds", {}) or {}
    row = {"neuron": neuron, "chain_idx": chain_idx,
           "n_frames": getattr(state, "n_frames", pd.NA),
           "peak_vram_gb": round(peak_vram, 3)}
    for p in _PHASES:                                  # always all 9, NA if a phase didn't run
        row[f"t_{p.split()[0]}"] = round(ps[p], 3) if p in ps else pd.NA
    for k in ("jpeg_load", "propagate_only"):
        row[f"t_{k}"] = round(sub[k], 3) if k in sub else pd.NA
    row["t_total"] = round(sum(ps.values()), 3)
    path = Path(output_root) / "_timing.csv"
    pd.DataFrame([row], columns=list(row)).to_csv(
        path, mode="a", header=not path.exists(), index=False)

# =============================================================================
# IO timing accumulator
# =============================================================================
# Full-file rewrites (the per-chain manifest breadcrumb especially) are pure
# overhead that scales with manifest size, not segmentation work. Track wall-clock
# + count per file so a run can say whether the breadcrumbing is worth throttling.
_IO_STATS: "OrderedDict[str, list]" = OrderedDict()   # filename -> [total_seconds, n_writes]


def _reset_io_stats() -> None:
    _IO_STATS.clear()


def _io_summary() -> str:
    if not _IO_STATS:
        return "[batch] io: no csv rewrites"
    parts = [f"{name} {secs:.2f}s/{n}" for name, (secs, n) in _IO_STATS.items()]
    total = sum(secs for secs, _ in _IO_STATS.values())
    n_all = sum(n for _, n in _IO_STATS.values())
    return (f"[batch] io: {total:.2f}s over {n_all} csv rewrites  "
            f"({', '.join(parts)})")

# =============================================================================
# Manifest ledger
# =============================================================================

def load_or_init_manifest(
    manifest_path: Path,
    all_chains: Sequence[Tuple[str, int, dict]],
) -> pd.DataFrame:
    """Load an existing manifest, or seed a fresh one with every chain `pending`.

    On resume, existing rows are kept as-is (their status drives skip/re-run);
    any chain present in `all_chains` but missing from the manifest is appended
    as `pending` (e.g. chains.json grew since the last run).
    """
    seed = pd.DataFrame(
        [{"neuron": n, "chain_idx": i, "status": PENDING,
          "n_frames": pd.NA, "n_flagged": pd.NA, "n_intervene": pd.NA,
          "flag_rate": pd.NA, "anchor_frame_idx": pd.NA,
          "anchor_passed": pd.NA, "anchor_reasons": "", "anchor_contained": pd.NA,
          "anchor_lcc": pd.NA, "anchor_area_frac": pd.NA,
          "error": "", "updated_at": _now()}
         for (n, i, _) in all_chains],
        columns=MANIFEST_COLUMNS,
    )

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return seed

    existing = pd.read_csv(manifest_path)
    for col in ("status", "error"):
        if col in existing.columns:
            existing[col] = existing[col].astype("object")
    
    # Append any new (neuron, chain_idx) not already tracked.
    have = set(zip(existing["neuron"], existing["chain_idx"]))
    new_rows = seed[~seed.apply(lambda r: (r["neuron"], r["chain_idx"]) in have, axis=1)]
    if not new_rows.empty:
        existing = pd.concat([existing, new_rows], ignore_index=True)
    # reindex (not [MANIFEST_COLUMNS]) so a manifest written by an older batch.py
    # without the anchor_* columns loads cleanly: missing columns come back as NA
    # rather than raising KeyError.
    return existing.reindex(columns=MANIFEST_COLUMNS)


def _clean_outputs(
    output_root: Path,
    all_chains: Sequence[Tuple[str, int, dict]],
    *,
    full: bool,
) -> None:
    """Remove prior on-disk artifacts so the next run starts from scratch.

    full=True  -> nuke the entire output_root (manifest, triage, timing, every
                  neuron dir). Total reset.
    full=False -> scoped: delete only the in-scope chains' dirs and drop their
                  rows from the top-level CSVs, leaving other neurons' finished
                  work untouched. (_triage.csv is rebuilt from scratch each run,
                  so it doesn't need pruning.)

    Unlike `force` (which re-runs but lets save_masks overwrite in place, leaving
    orphan PNGs from any chain whose frame coverage shrank), this deletes first,
    so QC never re-scores a stale mask.
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return

    if full:
        shutil.rmtree(output_root, ignore_errors=True)
        print(f"[batch] clean: removed {output_root}")
        return

    scoped = {(n, i) for (n, i, _) in all_chains}
    for neuron, idx, _ in all_chains:
        shutil.rmtree(output_root / neuron / f"chain_{idx:02d}", ignore_errors=True)
    # drop in-scope rows so they re-seed as `pending` (load_or_init re-adds them)
    for name in ("_manifest.csv", "_timing.csv"):
        p = output_root / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if {"neuron", "chain_idx"}.issubset(df.columns):
            keep = ~df.apply(lambda r: (r["neuron"], int(r["chain_idx"])) in scoped, axis=1)
            _atomic_write_csv(df[keep], p)
    print(f"[batch] clean: reset {len(scoped)} chains "
          f"in {sorted({n for n, _, _ in all_chains})}")


def _row_mask(manifest: pd.DataFrame, neuron: str, chain_idx: int):
    return (manifest["neuron"] == neuron) & (manifest["chain_idx"] == chain_idx)


def _status_of(manifest: pd.DataFrame, neuron: str, chain_idx: int) -> str:
    sub = manifest.loc[_row_mask(manifest, neuron, chain_idx), "status"]
    return str(sub.iloc[0]) if len(sub) else PENDING


def _update_row(manifest: pd.DataFrame, neuron: str, chain_idx: int, **fields) -> None:
    """In-place update of a manifest row. Always stamps updated_at."""
    mask = _row_mask(manifest, neuron, chain_idx)
    fields.setdefault("updated_at", _now())
    for col, val in fields.items():
        manifest.loc[mask, col] = val


def _should_run(status: str, retry_failed: bool, force: bool) -> bool:
    """Resume policy.

    - force            -> always run.
    - done / flagged   -> skip (ran to completion; flagged is handled via triage,
                          not by re-running).
    - failed           -> run only if retry_failed (failures are often transient,
                          e.g. OOM; default on).
    - pending          -> run.
    - running          -> run. A `running` row means the process died mid-chain
                          last time; re-run it cleanly.
    """
    if force:
        return True
    if status in COMPLETE_STATUSES:
        return False
    if status == FAILED:
        return retry_failed
    return True   # pending / running / anything unrecognized


def _should_tier2_rerun(status: str, cfg_chain_crop: bool, tier2_on_flagged: bool,
                        *, tier2_all: bool = False) -> bool:
    """Should a just-run chain get a second, tier-2 (per-chain crop) pass?

    Requires the first pass to have run in _sam, NOT already tier-2 (`not cfg_chain_crop`):
    a chain whose config already enabled chain_crop got tier-2 on its only pass, and the
    chain to have completed a _sam pass (`status in {done, flagged}`; failed/pending never
    re-run). Then:
      - ``tier2_all`` (default off): re-run EVERY completed chain, the "tier-2 everywhere"
        test mode. Pairs with chain_crop_from_mask so each chain's crop is sized from its own
        _sam mask bbox (clip-fixed), not the centerline. ~2x compute per chain.
      - else ``tier2_on_flagged``: re-run only the QC-FLAGGED chains (the default auto
        second-pass); clean `done` chains stay _sam.

    Tier-2's own fallback (chain_crop_fallback) reverts a chain with a poor crop
    anchor to the plain _sam path, so this second pass is regression-free. The win is
    landing chains in _pcrop: higher-res propagation for real drift, and a crisp
    manual-paint surface when the human opens the chain in the GUI.

    NB this fires only in the SAME invocation, right after the first pass. A chain already
    `done`/`flagged` on disk from a prior run is skipped by `_should_run` (both in
    COMPLETE_STATUSES), so it is never re-run twice; to upgrade a pre-existing backlog to
    tier-2, re-run those chains with `force=True`.
    """
    if cfg_chain_crop or status not in (DONE, FLAGGED):
        return False
    if tier2_all:
        return True
    return bool(tier2_on_flagged) and (status == FLAGGED)


# =============================================================================
# Per-chain run: THE wire-in point
# =============================================================================
# Everything above/below is plumbing that doesn't care how a chain runs. This is
# the *only* place that touches pipeline.run_chain. If the real signature drifts,
# fix it here and nowhere else.

@dataclass
class Session:
    """Built-once, reused-for-every-chain handles. Mirrors run_aval.py setup."""
    image_predictor: Any
    video_predictor: Any
    annotate_df: pd.DataFrame      # has x_tif / y_tif columns (transform applied)
    chains: Sequence[dict]
    frame_store: Any = None        # pipeline.FrameStore; None -> target-worm tif stack.
                                   # Set for a cross-worm run (e.g. SEM-Dauer 1 GT PNGs).


def _run_chain_once(session: Session, cfg: PipelineConfig, neuron: str,
                    chain_idx: int, chain: dict) -> ChainState:
    """One pipeline.run_chain call with the batch's standard wiring. Factored out so
    the tier-2 second pass can re-invoke it with a chain_crop override."""
    state = ChainState(neuron=neuron, chain_idx=chain_idx, config=cfg)
    return pipeline.run_chain(
        state,
        on_video_phase=diagnostics.cleanup_vram,
        image_predictor=session.image_predictor,
        video_predictor=session.video_predictor,
        annotate_df=session.annotate_df,
        chain=chain,                 # a single chain dict; enumerate_chains already indexed it
        frame_store=session.frame_store,   # None -> tif stack; GtFrameStore for SEM-Dauer 1
    )


def _run_one_chain(
    session: Session,
    cfg: PipelineConfig,
    neuron: str,
    chain_idx: int,
    chain: dict,
    chain_dir: Path,
    *,
    tier2_on_flagged: bool = True,
    tier2_all: bool = False,
) -> ChainState:
    """Run a single chain to completion and return its populated ChainState.

    This is the one place that calls pipeline.run_chain. run_chain gets the
    built predictors + annotate_df and a ChainState carrying (neuron, chain_idx,
    config); it runs all 9 phases (incl. run_qc), writes masks/ + qc.csv under
    the chain dir derived from cfg.output_root, sets state.status to done/flagged,
    and returns the populated state. Validated against run_aval.py.
    on_video_phase=cleanup_vram reclaims VRAM between the image and
    video phases (run_chain owns reset_predictor() internally).

    If the first (_sam) pass is FLAGGED and
    `tier2_on_flagged` is set, re-run the chain ONCE with chain_crop=True. We keep the
    tier-2 result unconditionally: its fallback already reverts a poor crop anchor
    to _sam, so the second pass never regresses and a kept-tier-2 chain lands in
    _pcrop (crisp GUI paint). The second run's save_masks overwrites the first pass's
    PNGs in place (same z-range/filenames), so there are no orphans. See _should_tier2_rerun
    for the precise trigger and the once-per-chain / backlog-upgrade semantics.
    """
    state = _run_chain_once(session, cfg, neuron, chain_idx, chain)
    if _should_tier2_rerun(getattr(state, "status", ""), cfg.chain_crop, tier2_on_flagged,
                           tier2_all=tier2_all):
        why = "all-chains mode" if tier2_all else "flagged in _sam"
        print(f"[batch] tier-2 re-run ({why}) {neuron}/chain_{chain_idx:02d}")
        diagnostics.cleanup_vram()                       # reclaim before the second pass
        state = _run_chain_once(session, replace(cfg, chain_crop=True),
                                neuron, chain_idx, chain)
        kept = ("fell back to _sam" if getattr(state, "fell_back_to_sam", False)
                else "kept tier-2 (_pcrop)")
        print(f"[batch] tier-2 re-run done {neuron}/chain_{chain_idx:02d}: "
              f"{kept}, status={getattr(state, 'status', '?')}")
    save_state(state, chain_dir / "state.json")
    return state


def _manifest_fields_from_state(state: ChainState) -> dict:
    """Pull the manifest summary columns off a finished ChainState.
    QC summary: n_frames / n_flagged / n_intervene / flag_rate.
    Anchor verdict: state.anchor_score (a plain dict; see score_anchor).
    """
    qs = getattr(state, "qc_summary", None) or {}
    a = getattr(state, "anchor_score", None) or {}
    contained = a.get("contained", None)
    return {
        "status": getattr(state, "status", DONE),
        "n_frames": qs.get("n_frames", pd.NA),
        "n_flagged": qs.get("n_flagged", pd.NA),
        "n_intervene": qs.get("n_intervene", pd.NA),
        "flag_rate": qs.get("flag_rate", pd.NA),
        "anchor_frame_idx": getattr(state, "anchor_frame_idx", pd.NA),
        # anchor gate. anchor_score is populated even on the empty-mask early-flag
        # path (it's scored before box_from_mask), so flagged-at-anchor chains
        # still get a verdict row here.
        "anchor_passed": a.get("passed", pd.NA),
        "anchor_reasons": ",".join(a.get("reasons", [])) if a else "",
        "anchor_contained": pd.NA if contained is None else contained,
        "anchor_lcc": a.get("largest_cc_frac", pd.NA),
        "anchor_area_frac": a.get("area_frac", pd.NA),
        "error": "",
    }


# =============================================================================
# Cross-chain triage rollup
# =============================================================================

def _reasons_for_row(row: pd.Series) -> str:
    """Short human-scannable tag string for a flagged frame.
    Mirrors qc.show_flagged's reason tags so the CSV reads the same as the viewer.
    """
    out = []
    if row.get("skeleton_contained") is False:
        out.append("noskel")
    ar = row.get("area_ratio")
    if pd.notna(ar) and not (0.5 <= ar <= 2.0):
        out.append(f"area x{ar:.1f}")
    ti = row.get("temporal_iou")
    if pd.notna(ti) and ti < 0.3:
        out.append(f"tIoU {ti:.2f}")
    pi = row.get("pred_iou")
    if pd.notna(pi) and pi < 0.5:
        out.append(f"pIoU {pi:.2f}")
    return " ".join(out)


def build_triage_queue(output_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every chain's flagged frames into output/_triage.csv.

    Reads each chain's on-disk qc.csv (known schema from qc.compute_metrics:
    indexed by z, with flag / flag_count / area_ratio / temporal_iou /
    skeleton_contained / pred_iou). Reading the artifact keeps this decoupled
    from in-memory ChainState internals: the filesystem is the index.
    """
    output_root = Path(output_root)
    frames: List[pd.DataFrame] = []

    for _, m in manifest.iterrows():
        neuron, idx = m["neuron"], int(m["chain_idx"])
        qc_csv = output_root / neuron / f"chain_{idx:02d}" / "qc.csv"
        if not qc_csv.exists():
            continue
        
        df = pd.read_csv(qc_csv)
        # surface only the queue frames (flag_count >= qc_triage_min_signals,
        # written by run_qc as the `queue` column). Fall back to `intervene`, then
        # `flag`, so a qc.csv written before the queue column still rolls up sensibly
        # (an existing run rebuilds straight to the intervene set, no re-segmentation).
        if "queue" in df.columns:
            sel = df["queue"] == True            # noqa: E712
        elif "intervene" in df.columns:
            sel = df["intervene"] == True        # noqa: E712
        elif "flag" in df.columns:
            sel = df["flag"] == True             # noqa: E712
        else:
            continue
        flagged = df[sel].copy()
        if flagged.empty:
            continue
        flagged.insert(0, "neuron", neuron)
        flagged.insert(1, "chain_idx", idx)
        flagged["reasons"] = flagged.apply(_reasons_for_row, axis=1)
        frames.append(flagged)

    if not frames:
        triage = pd.DataFrame(columns=["neuron", "chain_idx", "z", "flag_count",
                                       "intervene", "reasons"])
    else:
        triage = pd.concat(frames, ignore_index=True)
        # sort worst-first so the human clears the most-broken chains first
        triage = triage.sort_values(
            ["flag_count", "neuron", "chain_idx", "z"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

    _atomic_write_csv(triage, output_root / "_triage.csv")
    print(f"[batch] triage queue: {len(triage)} queued frames "
          f"-> {output_root / '_triage.csv'}")
    return triage


# =============================================================================
# Main batch loop
# =============================================================================

def run_batch(
    session: Session,
    cfg: PipelineConfig,
    output_root: Path,
    *,
    neurons: Optional[Sequence[str]] = None,
    retry_failed: bool = True,
    force: bool = False,
    clean: bool = False,         # True = delete prior outputs before running.
                                 #   full reset if neurons is None; else only those neurons.
    gif_mode: str = "flagged",   # "off" | "flagged" | "all": per-chain overlay gifs via review
    tier2_on_flagged: bool = True,  # re-run a flagged _sam chain
                                    # once with tier-2 crop (regression-free via fallback).
                                    # Set False to keep the legacy single-pass _sam behaviour.
    tier2_all: bool = False,        # "tier-2 everywhere" test mode: re-run EVERY completed
                                    # chain as tier-2 (pair with cfg.chain_crop_from_mask for
                                    # the clip-fixed, mask-sized crop). ~2x compute per chain.
) -> pd.DataFrame:
    """Run every (selected) chain, recording status to the manifest as it goes.

    Crash-safe: the manifest is rewritten after every chain, and a chain is
    marked `running` *before* it starts, so an interrupted run is visible and
    gets retried on the next invocation.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _reset_io_stats()
    manifest_path = output_root / "_manifest.csv"

    all_chains = enumerate_chains(session.chains, neurons)
    if clean:
        _clean_outputs(output_root, all_chains, full=(neurons is None))
    manifest = load_or_init_manifest(manifest_path, all_chains)
    _atomic_write_csv(manifest, manifest_path)

    print(f"[batch] {len(all_chains)} chains in scope "
          f"({len(set(n for n, _, _ in all_chains))} neurons)")

    ran = skipped = failed = 0
    for n_i, (neuron, idx, chain) in enumerate(all_chains, 1):
        status = _status_of(manifest, neuron, idx)
        tag = f"{neuron}/chain_{idx:02d}  [{n_i}/{len(all_chains)}]"

        if not _should_run(status, retry_failed, force):
            skipped += 1
            print(f"[batch] skip   {tag}  (status={status})")
            continue

        chain_dir = output_root / neuron / f"chain_{idx:02d}"
        chain_dir.mkdir(parents=True, exist_ok=True)

        # breadcrumb: if we die mid-chain, this row stays `running` and is retried
        # CONTINGENCY FOR SUDDEN UNPLUG OF DRIVE
        _update_row(manifest, neuron, idx, status=RUNNING, error="")
        _atomic_write_csv(manifest, manifest_path)
        print(f"[batch] run    {tag}")

        try:
            diagnostics.reset_peak_vram()
            state = _run_one_chain(session, cfg, neuron, idx, chain, chain_dir,
                                   tier2_on_flagged=tier2_on_flagged, tier2_all=tier2_all)
            try:
                _append_timing(output_root, neuron, idx, state, diagnostics.peak_vram_gb())
            except Exception as e:
                print(f"[batch] timing skipped {tag}: {e}")   # never let telemetry kill a chain
            
            fields = _manifest_fields_from_state(state)
            _update_row(manifest, neuron, idx, **fields)
            ran += 1
            # overlay gif/mp4 for eyeballing: review reads the chain's on-disk masks.
            #   "off"     -> never; "flagged" -> only chains QC flagged; "all" -> every chain.
            # "flagged" keeps a full overnight run from gifing every clean chain.
            is_flagged = fields["status"] == FLAGGED
            if gif_mode == "all" or (gif_mode == "flagged" and is_flagged):
                suffix = "_flagged" if is_flagged else ""
                try:
                    review.to_gif(chain_dir, chain_dir / f"{neuron}_chain{idx:02d}{suffix}.gif")
                    review.to_mp4(chain_dir, output_root / "mp4" / f"{neuron}_chain{idx:02d}.mp4")
                except Exception as e:
                    print(f"[batch] gif skipped {tag}: {e}")
        except Exception as e:           # one bad chain must not kill the batch
            failed += 1
            _update_row(manifest, neuron, idx, status=FAILED,
                        error=f"{type(e).__name__}: {e}")
            print(f"[batch] FAILED {tag}: {e}")
            traceback.print_exc()
        finally:
            _atomic_write_csv(manifest, manifest_path)
            # free VRAM between chains on long overnight runs
            diagnostics.cleanup_vram()

    print(f"[batch] done. ran={ran} skipped={skipped} failed={failed}")
    build_triage_queue(output_root, manifest)
    print(_io_summary())
    return manifest


# =============================================================================
# Bootstrap  (mirror run_aval.py's setup, then loop instead of single-chain)
# =============================================================================

def _build_session(cfg: PipelineConfig) -> Session:
    """One-time session setup. Lift this almost verbatim from run_aval.py."""
    # Predictors built ONCE, above the driver (cfg.model_size consumed here, not
    # inside run_chain).
    image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image",
                                                image_size=cfg.image_size)
    video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video",
                                               image_size=cfg.image_size)
    diagnostics.snapshot("after model load")

    # annotate_df: CATMAID pull (or cached CSV), then apply the stack->tif affine.
    # annotate_df: cached CSV (default). To refresh live from CATMAID instead:
    #   from sam2_utils import catmaid
    #   annotate_df = catmaid.fetch_all_annotations(catmaid.Catmaid())
    annotate_df = pd.read_csv(CSV_PATH)
    xy_tif = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"] = xy_tif[:, 0]
    annotate_df["y_tif"] = xy_tif[:, 1]

    # chains.json
    with open(CHAINS_PATH) as f:
        chains = json.load(f)

    return Session(image_predictor, video_predictor, annotate_df, chains)


def _build_gt_session(cfg: PipelineConfig,
                      neurons: Optional[Sequence[str]] = None,
                      neuron_limit: Optional[int] = None) -> Session:
    """Session for a SEM-Dauer 1 (cross-worm GT) run.

    Same predictors as the target worm, but the dataset seams come from
    `eval.gt_dataset`: annotate_df with x_tif/y_tif from the per-section registration,
    a configurable chain subset (`neurons` / `neuron_limit`), and a GtFrameStore
    (per-slice PNG EM). Chain cell_names are normalized (strips brackets + trailing
    !/?, also dodges the Windows mkdir-on-'?' bug)."""
    from eval import gt_dataset
    from sam2_utils.skeletons import normalize_name

    image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image",
                                                image_size=cfg.image_size)
    video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video",
                                               image_size=cfg.image_size)
    diagnostics.snapshot("after model load")

    annotate_df, chains, frame_store = gt_dataset.build_gt_session_inputs(
        neurons, neuron_limit)
    for c in chains:
        c["cell_name"] = normalize_name(c["cell_name"])
    print(f"[batch] SEM-Dauer 1: {len(chains)} chains over "
          f"{len({c['cell_name'] for c in chains})} neurons; EM={frame_store.em_dir}")
    return Session(image_predictor, video_predictor, annotate_df, chains, frame_store)


def write_run_meta(output_root: Path, *, preset: str, cfg: PipelineConfig,
                   neurons: Optional[Sequence[str]], gif_mode: str,
                   tier2_on_flagged: bool, tier2_all: bool, session: "Session") -> None:
    """Write ``_run_meta.json``: full provenance for a run we cannot watch live.

    A cluster run is fire-and-forget (no live console, Duo-gated login), so persist
    everything needed to reconstruct what actually executed for post-hoc comparison:
    the preset and resolved resolution/tier-2 knobs, the git commit + dirty flag, host,
    the argv, the neuron scope, and the video predictor's ACTUAL ``image_size`` (so the
    bigimg variant records the size it truly built at, not just what was requested). Pairs
    with the per-chain ``state.json`` (phase timings, ``fell_back_to_sam``, QC summary) and
    ``_manifest.csv`` / ``_timing.csv`` already written per run.
    """
    import sys
    import platform
    import subprocess
    from dataclasses import asdict, is_dataclass

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    def _git(*a: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *a], cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    meta = {
        "preset": preset,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "argv": sys.argv,
        "resolution": {
            "scale": cfg.scale,
            "save_downscale": cfg.save_downscale,
            "image_size_requested": cfg.image_size,
            "image_size_actual": getattr(session.video_predictor, "image_size", None),
        },
        "tier2": {
            "tier2_on_flagged": tier2_on_flagged,
            "tier2_all": tier2_all,
            "chain_crop_min_image_score": cfg.chain_crop_min_image_score,
            "chain_crop_from_mask": cfg.chain_crop_from_mask,
        },
        "model_size": cfg.model_size,
        "gif_mode": gif_mode,
        "neurons": list(neurons) if neurons else None,
        "n_neurons": len(neurons) if neurons else None,
        "pipeline_config": asdict(cfg) if is_dataclass(cfg) else None,
    }
    path = output_root / "_run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[batch] wrote run provenance -> {path} "
          f"(image_size actual={meta['resolution']['image_size_actual']})")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Headless batch runner. Pick a run config with --preset "
                    f"({'/'.join(sorted(presets.PRESETS))}); any flag below overrides it. "
                    "Presets live in sam2_utils/presets.py.")
    ap.add_argument("--preset", choices=sorted(presets.PRESETS), default="original",
                    help="run configuration: 'eval' = SEM-Dauer 1 GT, 'original' = target worm")
    ap.add_argument("--neurons", nargs="*", default=None,
                    help="explicit neuron allow-list (overrides the preset's default)")
    ap.add_argument("--neuron-limit", type=int, default=None,
                    help="run only the first N neurons (by sorted name), a quick subset")
    ap.add_argument("--all", action="store_true",
                    help="GT: run EVERY neuron (~9766 chains). Required to opt in to a full run.")
    ap.add_argument("--clean", action="store_true", help="wipe prior outputs first")
    ap.add_argument("--output-root", type=Path, default=None, help="override the preset output root")
    ap.add_argument("--frames-root", type=Path, default=None, help="override the preset frames root")
    ap.add_argument("--model-size", default=None, help="override the preset model size")
    ap.add_argument("--gif-mode", choices=["off", "flagged", "all"], default=None,
                    help="override the preset gif mode")
    ap.add_argument("--no-tier2", action="store_true",
                    help="disable the tier-2 second pass entirely (overrides the preset)")
    ap.add_argument("--postprocess", dest="postprocess", action="store_true", default=None,
                    help="force mask post-processing ON (overrides the preset; A/B with --no-postprocess)")
    ap.add_argument("--no-postprocess", dest="postprocess", action="store_false",
                    help="force mask post-processing OFF (overrides the preset)")
    args = ap.parse_args()

    p = presets.get_preset(args.preset)
    pipe = dict(p["pipeline"])
    if args.model_size:
        pipe["model_size"] = args.model_size
    if args.postprocess is not None:
        pipe["postprocess_masks"] = args.postprocess
    cfg = PipelineConfig(**pipe,
                         output_root=args.output_root or p["output_root"],
                         frames_root=args.frames_root or p["frames_root"])
    neurons = args.neurons if args.neurons else p["neurons"]
    clean = args.clean or p["clean"]
    gif_mode = args.gif_mode or p["gif_mode"]
    tier2_flagged = False if args.no_tier2 else p["tier2_on_flagged"]
    tier2_all = False if args.no_tier2 else p["tier2_all"]

    if p["dataset"] == "sem-dauer-1":
        # Guard the expensive full run: require an explicit scope (9766 chains × slow
        # full-res PNG frame-prep is days of compute, don't do it by accident).
        if not neurons and args.neuron_limit is None and not args.all:
            ap.error("preset 'eval' (SEM-Dauer 1) needs an explicit scope: pass "
                     "--neurons NAME ..., --neuron-limit N, or --all.")
        session = _build_gt_session(cfg, neurons, args.neuron_limit)
        # scope run_batch (clean + enumerate) to exactly the chains we loaded
        resolved = sorted({c["cell_name"] for c in session.chains})
        write_run_meta(Path(cfg.output_root), preset=args.preset, cfg=cfg, neurons=resolved,
                       gif_mode=gif_mode, tier2_on_flagged=tier2_flagged,
                       tier2_all=tier2_all, session=session)
        run_batch(session, cfg, Path(cfg.output_root), neurons=resolved, clean=clean,
                  gif_mode=gif_mode, tier2_on_flagged=tier2_flagged, tier2_all=tier2_all)
        return

    # --- target worm ---
    session = _build_session(cfg)
    write_run_meta(Path(cfg.output_root), preset=args.preset, cfg=cfg, neurons=neurons,
                   gif_mode=gif_mode, tier2_on_flagged=tier2_flagged,
                   tier2_all=tier2_all, session=session)
    run_batch(session, cfg, Path(cfg.output_root), neurons=neurons, clean=clean,
              gif_mode=gif_mode, tier2_on_flagged=tier2_flagged, tier2_all=tier2_all)


if __name__ == "__main__":
    main()