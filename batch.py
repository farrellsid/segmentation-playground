"""
batch.py — headless batch runner + resume (milestone 3 scaffold).

This is run_aval.py generalized into a loop. Same session setup (predictors,
annotate_df, chains built once), then run *every* chain unattended, recording
status to a manifest ledger as it goes, and rolling the per-chain QC flags up
into one cross-chain triage queue.

What M3 is and isn't
--------------------
IS:  run all chains overnight, survive crashes (resume from the manifest),
     never recompute a finished chain, and produce `_triage.csv` so you can
     measure the auto-flag rate across the whole dataset before building the GUI.
ISN'T: mid-propagation halt-and-re-prompt. That's the `propagate` generator
     restructure, coupled to the napari GUI, and lives in M4. This runner treats
     each chain as a single atomic `run_chain` call: run it, record what QC
     flagged, move on. Resist wiring interventions in here.

Storage (PIPELINE_CONTEXT §3e)
------------------------------
    output/
      _manifest.csv                 # every chain x status — drives batch + resume
      _triage.csv                   # flagged frames across all chains — feeds the GUI
      <neuron>/chain_<idx:02d>/
        state.json                  # ChainState (save_state/load_state)
        qc.csv                      # per-frame metrics (indexed by catmaid_z)
        masks/mask_<catmaid_z:04d>.png

Usage
-----
    python batch.py                 # runs main(): session setup, then run_batch
    # or import run_batch / build_triage_queue from a notebook for inspection.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

# Project imports. `pipeline` is the M1/M2 library at the repo root; the
# sam2_utils pieces are the stable helpers.
import pipeline
from pipeline import ChainState, PipelineConfig, save_state  # load_state if resuming state
from sam2_utils import setup, alignment, diagnostics, review

# DATA PATHS
CSV_PATH    = Path(r"D:\Zhen Lab\SAM2 Segmentation\segmentation-playground\data\aggregate_data_pv.csv")
CHAINS_PATH = Path(r"D:\Zhen Lab\SAM2 Segmentation\segmentation-playground\data\chains.json")

OUTPUT_ROOT = Path(r"E:\ZhenLab\Data\output_masks\test2_single")
FRAMES_ROOT = Path(r"E:\ZhenLab\Data")     # SAM2 JPEG frame folders go here

# Simple Neurons list
all_neurons = ['AIAR', 'RIS', 'GLRVR', 'PLNR', 'SAADL', 'AVBR', 'PVQL', 'URADR',
            'AVBL', 'RIBL', 'SAAVR', 'RMED', 'PLNL', 'AVHL', 'AVM',
            'SDQL', 'PVWL_or_R_3', 'RMGL', 'RMHL', 'SMDDL', 'AINL', 'PVPL', 'RMDL',
            'RMFL', 'AVDR', 'URYDR', 'SMBVL', 'ALA', 'RICL', 'SMDVL', 'RIGL', 'SABD',
            'ADAL', 'AIAL', 'AVAR','FLPR', 'URAVL', 'RMEL', 'URYVL', 'URBL', 'AIZL',
            'AVJR', 'URBR', 'RIML', 'AIMR', 'ALNR', 'PVDL', 'SMBDL','SAAVL', 'ALMR',
            'RIAL', 'VB1', 'SDQR','PVPR', 'SIADR', 'AIZR', 'AIYR', 'RIR','PVCL',
            'PVR', 'SIBDL', 'RMDVR', 'RIAR','RID', 'SMDVR','AUAL', 'PVWL_or_R_1',
            'RICR', 'AVFL', 'AIBR', 'BDUL', 'SIADL', 'AVFR', 'SMDDR', 'PVT',
            'ALML', 'RMER', 'PVQR','RIPL', 'RMGR', 'AVHR', 'RIPR', 'RMHR', 'RMFR',
            'PVWL_or_R_2', 'AIYL', 'BDUR', 'RIVR', 'AVKR', 'RMEV', 'RMDR', 'AIML',
            'AVER', 'RIFR', 'SIBDR', 'RIMR', 'RMDDR', 'AVKL', 'RIBR', 'CANR',
            'DVA', 'SIAVR', 'AVJL', 'RIFL', 'SAADR', 'AIBL', 'URAVR',
            'AVEL', 'ADAR', 'AINR', 'SIBVL', 'RMDVL', 'SIAVL', 'AVL', 'AUAR',
            'SMBVR', 'DVC', 'URADL', 'PVCR', 'URYVR', 'AVAL', 'RMDDL', 'SIBVR',
            'PVDR', 'URYDL', 'ALNL', 'FLPL', 'AVDL', 'SABVL', 'RIH', 'RIGR', 'RIVL', 'SMBDR']

key_neurons = ['AIYR', 'AIYL', 'AIAR', 'AIAL', 'AIZL', 'AIZR', 'AIBL', 'AIBR', 'URAVR', 'URAVL', 'URADL', 'URADR', 'RIH', 'RIPL', 'RIPR']


# RUN KNOBS (edit per launch)z
NEURONS: Optional[Sequence[str]] =  key_neurons[1:6] # e.g. ["AVAL", "AVAR"]; None = all objects

CLEAN = True                             # True = wipe prior outputs and start fresh

# Status vocabulary (matches ChainState.status / PIPELINE_CONTEXT §3b).
PENDING, RUNNING, DONE, FLAGGED, FAILED = (
    "pending", "running", "done", "flagged", "failed",
)
COMPLETE_STATUSES = {DONE, FLAGGED}      # ran to completion; don't re-run on resume

MANIFEST_COLUMNS = [
    "neuron", "chain_idx", "status",
    "n_frames", "n_flagged", "n_intervene", "flag_rate",
    "anchor_frame_idx",
    # anchor-quality gate verdict (M3.5 item 1), rolled up per chain so it sits
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
    can't corrupt the manifest — you either get the old file or the new one.
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

    chain_idx is the position *within that neuron's* chain list — the same index
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
# Full-file rewrites — the per-chain manifest breadcrumb especially — are pure
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
    # without the anchor_* columns loads cleanly — missing columns come back as NA
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
    orphan PNGs from any chain whose frame coverage shrank), this deletes first —
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


# =============================================================================
# Per-chain run — THE wire-in point
# =============================================================================
# Everything above/below is plumbing that doesn't care how a chain runs. This is
# the *only* place that touches pipeline.run_chain. If the real signature drifts,
# fix it here and nowhere else.

@dataclass
class Session:
    """Built-once, reused-for-every-chain handles. Mirrors run_aval.py setup."""
    image_predictor: Any
    video_predictor: Any
    annotate_df: pd.DataFrame      # has x_tif / y_tif columns (affine applied)
    chains: Sequence[dict]


def _run_one_chain(
    session: Session,
    cfg: PipelineConfig,
    neuron: str,
    chain_idx: int,
    chain: dict,
    chain_dir: Path,
) -> ChainState:
    """Run a single chain to completion and return its populated ChainState.

    TODO(NE?)[M3]: verify this run_chain() call against run_aval.py — arg list + out_dir.
    From the M1 bootstrap, run_chain gets the built predictors + annotate_df and
    a ChainState carrying (neuron, chain_idx, config). It derives the subchain
    from neuron+chain_idx, runs all 9 phases incl. run_qc, writes masks/ + qc.csv
    under chain_dir, sets state.status to done/flagged, and returns the state.
    Adjust the argument list to match what you actually wrote.
    """
    state = ChainState(neuron=neuron, chain_idx=chain_idx, config=cfg)

    # NOTE: reset_predictor() / cleanup_vram() between image and video phases is
    # done *inside* run_chain in the library version (the notebook did it inline).
    # If your run_chain doesn't, do it there, not here.
    state = pipeline.run_chain(
        state,
        on_video_phase=diagnostics.cleanup_vram,    # TODONE: verified cleanup is done in-function
        image_predictor=session.image_predictor,
        video_predictor=session.video_predictor,
        annotate_df=session.annotate_df,
        chain=chain,      # Checked: argument is chain, not chains. Only a single chain dictionary is passed. This assumes the correct chain is passed meaning the indexing of the chains is all done before calling i guess
        # TODONE: output directory is defined by cfg.py. out_dir is derived from that.
    )   
    save_state(state, chain_dir / "state.json")
    return state


def _manifest_fields_from_state(state: ChainState) -> dict:
    """Pull the manifest summary columns off a finished ChainState.
    QC summary per M2: n_frames / n_flagged / n_intervene / flag_rate.
    Anchor verdict per M3.5: state.anchor_score (a plain dict; see score_anchor).
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
    from in-memory ChainState internals — the filesystem is the index (§4).
    """
    output_root = Path(output_root)
    frames: List[pd.DataFrame] = []

    for _, m in manifest.iterrows():
        neuron, idx = m["neuron"], int(m["chain_idx"])
        qc_csv = output_root / neuron / f"chain_{idx:02d}" / "qc.csv"
        if not qc_csv.exists():
            continue
        df = pd.read_csv(qc_csv)
        if "flag" not in df.columns:
            continue
        flagged = df[df["flag"] == True].copy()   # noqa: E712 (pandas parses the CSV bool col back as bool)
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
    print(f"[batch] triage queue: {len(triage)} flagged frames "
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
        _update_row(manifest, neuron, idx, status=RUNNING, error="")
        _atomic_write_csv(manifest, manifest_path)
        print(f"[batch] run    {tag}")

        try:
            diagnostics.reset_peak_vram()
            state = _run_one_chain(session, cfg, neuron, idx, chain, chain_dir)
            try:
                _append_timing(output_root, neuron, idx, state, diagnostics.peak_vram_gb())
            except Exception as e:
                print(f"[batch] timing skipped {tag}: {e}")   # never let telemetry kill a chain
            
            fields = _manifest_fields_from_state(state)
            _update_row(manifest, neuron, idx, **fields)
            ran += 1
            # overlay gif for eyeballing — review reads the chain's on-disk masks.
            # default "flagged" so a full overnight run only gifs the chains worth looking at.
            if gif_mode == "all" or (gif_mode == "flagged" and fields["status"] == FLAGGED):
                try:
                    review.to_gif(chain_dir, chain_dir / f"{neuron}_chain{idx:02d}.gif")
                    review.to_mp4(chain_dir, OUTPUT_ROOT / "mp4" / f"{neuron}_chain{idx:02d}.mp4")
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
    image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image")
    video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video")
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


def main() -> None:
    cfg = PipelineConfig( # defaults; tune qc_* knobs here
        model_size="large",
        scale=8,
        save_downscale=8,        # canonical: == scale, no resample, no 2x skeleton bug
        k_max_neg=7,
        neg_radius=150,          # accepted but unused in M1 (see build_prompts docstring)
        box_margin=10,
        output_root=OUTPUT_ROOT,
        frames_root=FRAMES_ROOT,
        )                     
    output_root = Path(cfg.output_root)        # TODONE[M3]: confirmed attr names on PipelineConfig
    session = _build_session(cfg)
    run_batch(session, cfg, output_root, neurons=NEURONS, clean=CLEAN)


if __name__ == "__main__":
    main()