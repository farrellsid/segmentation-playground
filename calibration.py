"""
calibration.py — gold-set labeling + QC-threshold calibration (read-only).

Purpose
-------
A lightweight, human-in-the-loop pass that turns a finished batch run into a
small *verified ground-truth set*, so the QC error-detection signals can be
calibrated against truth instead of against themselves. It exists because of two
blind spots established in PIPELINE_CONTEXT §4/§7:

  1. Anchor contamination. The temporal signals (area_ratio, temporal_iou,
     centroid_jump) are measured relative to the previous frame, so if the
     image-mode anchor mask is wrong, every downstream flag is unattributable.
     => we review the anchor as a per-CHAIN gate; only trusted-anchor chains
        are eligible for propagation-threshold calibration.

  2. Silent errors. A stable-but-wrong mask propagates with clean area_ratio
     (~1) and high temporal_iou and therefore never flags. Those errors are not
     in the triage queue, so the queue alone can't estimate the false-negative
     rate. => we label a random sample of UN-flagged frames to estimate it.

NOT the M4 GUI. This is read-only: it renders saved masks and records human
verdicts. No point editing, no re-prompting, no re-segmentation. (The threshold
sweep re-scores existing masks via qc.compute_metrics; it never re-runs SAM2.)
Per PIPELINE_CONTEXT §4 there is exactly one *correction* tool (napari, M4); this
is a *labeling* tool and must stay that way — do not add click-to-edit here.

Inputs (produced by batch.py, on disk):
    output_root/
      _manifest.csv                 # which chains exist + their status
      <neuron>/chain_<idx:02d>/
        state.json                  # anchor_frame_idx / anchor_catmaid_z, frame_to_z
        qc.csv                      # per-frame signals + flag/intervene (key: z)
        masks/mask_<z:04d>.png
Output:
    output_root/_labels.csv         # the gold set (joins to qc.csv on neuron,chain_idx,z)

Workflow
--------
    wl = build_worklist(out, "AVAL")          # anchor + all flagged + sampled unflagged
    review_chain(out, "AVAL", 0, wl)          # render that chain's worklist (reuses review.py)
    record_chain_labels(wl, "AVAL", 0,        # label by exception
        anchor_ok=True,
        bad={1583: "bleed", 1590: "wrong_object"},
        reviewer="sf", out_csv=out/"_labels.csv")
    # ...repeat over a few dozen chains...
    print_report(score(out/"_labels.csv", out))          # precision / est. recall / FN rate
    sweep_thresholds(out/"_labels.csv", out, skeleton=annotate_df)   # offline param sweep
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# Verdict vocabulary. "ok" is implicit (label by exception); these tag *why* a
# mask is wrong, which is what makes the calibration diagnosable per error mode.
ERROR_TYPES = ("wrong_object", "under", "bleed", "fragmented", "missing", "other")

LABEL_COLS = [
    "neuron", "chain_idx", "z", "role", "verdict", "error_type",
    "reviewer", "ts", "notes",
]


# ---------------------------------------------------------------------------
# Loading the batch artifacts
# ---------------------------------------------------------------------------
def _chain_dir(output_root: Path, neuron: str, chain_idx: int) -> Path:
    return Path(output_root) / neuron / f"chain_{chain_idx:02d}"


def _load_qc(output_root: Path, neuron: str, chain_idx: int) -> pd.DataFrame:
    return pd.read_csv(_chain_dir(output_root, neuron, chain_idx) / "qc.csv")


def _anchor_z(output_root: Path, neuron: str, chain_idx: int) -> Optional[int]:
    """Resolve the anchor's catmaid_z from state.json (the only place it lives)."""
    p = _chain_dir(output_root, neuron, chain_idx) / "state.json"
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    if st.get("anchor_catmaid_z") is not None:
        return int(st["anchor_catmaid_z"])
    f2z, idx = st.get("frame_to_z"), st.get("anchor_frame_idx")
    if f2z is not None and idx is not None:
        # frame_to_z keys come back as strings from JSON
        return int(f2z.get(str(idx), f2z.get(idx)))
    return None


def _eligible_chains(output_root: Path, neuron: str) -> list[int]:
    """Chains worth reviewing: anything the batch actually finished."""
    man = pd.read_csv(Path(output_root) / "_manifest.csv")
    sel = man[(man.neuron == neuron) & man.status.isin(["done", "flagged"])]
    return sorted(int(c) for c in sel.chain_idx.unique())


# ---------------------------------------------------------------------------
# Stage A — build the review worklist
# ---------------------------------------------------------------------------
def build_worklist(
    output_root: str | Path,
    neuron: str,
    *,
    n_unflagged_sample: int = 8,
    seed: int = 0,
    chains: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per chain: the anchor frame + every flagged frame + a uniform random
    sample of un-flagged frames.

    Roles:
      anchor   — the seed; its verdict gates the whole chain (see module docstring)
      flagged  — qc flagged it: the precision set (label all of them)
      sampled  — qc called it clean: the recall set (estimates the FN rate)

    The un-flagged sample is *uniform* on purpose — a biased sample would bias the
    silent-error rate. Bump n_unflagged_sample for tighter FN estimates.
    """
    output_root = Path(output_root)
    rng = np.random.default_rng(seed)
    chains = list(chains) if chains is not None else _eligible_chains(output_root, neuron)

    rows: list[dict] = []
    for ci in chains:
        qc = _load_qc(output_root, neuron, ci)
        a_z = _anchor_z(output_root, neuron, ci)
        flagged = set(qc.loc[qc.flag == True, "z"].astype(int))          # noqa: E712
        unflagged = qc.loc[qc.flag != True, "z"].astype(int).tolist()    # noqa: E712
        if a_z in unflagged:
            unflagged.remove(a_z)                                         # anchor scored on its own row
        k = min(n_unflagged_sample, len(unflagged))
        sampled = set(rng.choice(unflagged, size=k, replace=False).tolist()) if k else set()

        for _, r in qc.iterrows():
            z = int(r.z)
            role = ("anchor" if z == a_z
                    else "flagged" if z in flagged
                    else "sampled" if z in sampled
                    else None)
            if role is None:
                continue
            rows.append({
                "neuron": neuron, "chain_idx": ci, "z": z, "role": role,
                # carry the signals so the reviewer sees context inline
                "flag": bool(r.flag), "intervene": bool(r.intervene),
                "skel_contained": r.skeleton_contained,
                "area_ratio": r.area_ratio, "temporal_iou": r.temporal_iou,
                "n_components": r.n_components, "logit_conf": r.get("logit_conf", np.nan),
            })
    wl = pd.DataFrame(rows)
    if len(wl):
        n = wl.groupby("role").size().to_dict()
        print(f"[worklist] {neuron}: {len(chains)} chains | "
              f"anchor={n.get('anchor',0)} flagged={n.get('flagged',0)} sampled={n.get('sampled',0)}")
    return wl


# ---------------------------------------------------------------------------
# Stage B — render (read-only) and record verdicts
# ---------------------------------------------------------------------------
def worklist_reviewdata(output_root: str | Path, neuron: str, chain_idx: int,
                        worklist: pd.DataFrame):
    """Load the chain via review.load_chain and return a ReviewData filtered to
    *only* the worklist frames.

    review.grid / video_viz.grid select frames from whatever is in
    `video_segments` (keyed by frame_idx), so we map worklist z -> frame_idx via
    `frame_to_z` and drop everything else. Returned object feeds straight into
    review.grid(...) or review.animate(...).
    """
    from dataclasses import replace
    from sam2_utils import review
    data = review.load_chain(_chain_dir(Path(output_root), neuron, chain_idx))
    z_to_frame = {int(z): idx for idx, z in data.frame_to_z.items()}
    wl = worklist[(worklist.neuron == neuron) & (worklist.chain_idx == chain_idx)]
    want_z = {int(z) for z in wl.z}
    keep = {z_to_frame[z] for z in want_z if z in z_to_frame}
    missing = sorted(want_z - set(z_to_frame))
    if missing:
        print(f"[review_chain] {len(missing)} worklist z not in frame_to_z, skipped: "
              f"{missing[:6]}{'...' if len(missing) > 6 else ''}")
    sub = replace(data, video_segments={i: m for i, m in data.video_segments.items()
                                        if i in keep})
    return sub


def review_chain(output_root: str | Path, neuron: str, chain_idx: int,
                 worklist: pd.DataFrame, *, cols: int = 4):
    """Print the worklist (with signal context) and render exactly those frames
    + masks as a contact sheet for eyeballing.

    Read-only: defers to review.grid so 'how a mask is shown' has one definition.
    Returns the Figure. For scrubbing an ambiguous chain instead of a grid:
        review.animate(worklist_reviewdata(out, neuron, ci, wl))
    """
    wl = worklist[(worklist.neuron == neuron) & (worklist.chain_idx == chain_idx)]
    print(wl.sort_values("z")[["z", "role", "flag", "intervene", "skel_contained",
                               "area_ratio", "temporal_iou", "n_components"]]
          .to_string(index=False))
    try:
        from sam2_utils import review
        sub = worklist_reviewdata(output_root, neuron, chain_idx, worklist)
        if not sub.video_segments:
            print("[review_chain] no worklist frames matched masks on disk.")
            return None
        # n >= kept-frame count so grid shows ALL worklist frames, not an
        # evenly-spaced subset of them.
        return review.grid(sub, n=len(sub.video_segments), cols=cols)
    except Exception as e:  # pragma: no cover - rendering is environment-dependent
        print(f"[review_chain] render unavailable ({e}); "
              f"open chain_{chain_idx:02d} with sam2_utils.review manually.")
        return None


def record_chain_labels(
    worklist: pd.DataFrame,
    neuron: str,
    chain_idx: int,
    *,
    anchor_ok: bool,
    bad: Optional[Mapping[int, str]] = None,
    reviewer: str,
    out_csv: str | Path,
    notes: str = "",
) -> pd.DataFrame:
    """Label by exception. Everything in this chain's worklist that is NOT in
    `bad` is recorded 'ok'; `bad` maps z -> error_type for the wrong ones.
    `anchor_ok` is the chain-level gate (the anchor's verdict).

    Appends rows to out_csv (idempotent per (neuron, chain_idx, z) — re-recording
    a chain overwrites its previous rows).
    """
    bad = {int(k): v for k, v in (bad or {}).items()}
    for et in bad.values():
        if et not in ERROR_TYPES:
            raise ValueError(f"unknown error_type {et!r}; allowed: {ERROR_TYPES}")
    wl = worklist[(worklist.neuron == neuron) & (worklist.chain_idx == chain_idx)]
    if not len(wl):
        raise ValueError(f"no worklist rows for {neuron} chain {chain_idx}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for _, r in wl.iterrows():
        z = int(r.z)
        is_bad = z in bad
        # the anchor verdict overrides: a bad anchor is a 'wrong' anchor row
        if r.role == "anchor":
            verdict = "ok" if anchor_ok else "wrong"
            et = "" if anchor_ok else bad.get(z, "other")
        else:
            verdict = "wrong" if is_bad else "ok"
            et = bad.get(z, "") if is_bad else ""
        out.append({
            "neuron": neuron, "chain_idx": chain_idx, "z": z, "role": r.role,
            "verdict": verdict, "error_type": et, "reviewer": reviewer, "ts": ts,
            "notes": notes,
        })
    new = pd.DataFrame(out, columns=LABEL_COLS)

    out_csv = Path(out_csv)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        mask = ~((old.neuron == neuron) & (old.chain_idx == chain_idx))
        new = pd.concat([old[mask], new], ignore_index=True)
    new.to_csv(out_csv, index=False)
    print(f"[labels] {neuron} chain {chain_idx}: anchor_ok={anchor_ok}, "
          f"{len(bad)} wrong / {len(wl)} reviewed -> {out_csv}")
    return new


# ---------------------------------------------------------------------------
# Stage C — score the labels against the QC signals
# ---------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    n_chains: int
    n_chains_bad_anchor: int
    # precision side (flagged frames on trusted-anchor chains)
    tp: int            # flagged & wrong
    fp: int            # flagged & ok
    precision: float
    # recall side (estimated from the uniform un-flagged sample)
    sampled_unflagged: int
    sampled_wrong: int
    silent_error_rate: float      # wrong / sampled  -> the FN blind spot
    est_recall: float
    by_error_type: dict


def score(labels_csv: str | Path, output_root: str | Path) -> CalibrationReport:
    """Join labels to qc.csv and compute precision (measured), the silent-error
    rate, and an estimated recall — restricted to trusted-anchor chains, since
    flags on bad-anchor chains are unattributable.
    """
    lab = pd.read_csv(labels_csv)
    output_root = Path(output_root)

    # chain-level anchor gate
    anchor = lab[lab.role == "anchor"][["neuron", "chain_idx", "verdict"]]
    bad_anchor = set(map(tuple, anchor[anchor.verdict == "wrong"][["neuron", "chain_idx"]].values))
    trusted = lab[~lab.set_index(["neuron", "chain_idx"]).index.isin(bad_anchor)].copy()

    # attach the actual qc flag per labeled frame (authoritative source = qc.csv)
    def _flag(row):
        qc = _load_qc(output_root, row.neuron, int(row.chain_idx))
        hit = qc.loc[qc.z == row.z, "flag"]
        return bool(hit.iloc[0]) if len(hit) else False
    trusted["qc_flag"] = trusted.apply(_flag, axis=1)
    trusted["wrong"] = trusted.verdict == "wrong"

    flagged = trusted[trusted.role == "flagged"]
    tp = int((flagged.wrong).sum())
    fp = int((~flagged.wrong).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    smp = trusted[trusted.role == "sampled"]
    n_smp = len(smp)
    wrong_smp = int(smp.wrong.sum())
    silent_rate = wrong_smp / n_smp if n_smp else float("nan")

    # Estimated FN: extrapolate the silent-error rate over all un-flagged frames
    # on trusted chains, then est_recall = TP / (TP + est_FN).
    n_unflagged_total = 0
    for (neuron, ci) in {(n, int(c)) for n, c in trusted[["neuron", "chain_idx"]].values}:
        qc = _load_qc(output_root, neuron, ci)
        n_unflagged_total += int((qc.flag != True).sum())                # noqa: E712
    est_fn = silent_rate * n_unflagged_total if n_smp else float("nan")
    est_recall = tp / (tp + est_fn) if (tp + (est_fn or 0)) else float("nan")

    by_type = (trusted[trusted.wrong].error_type.value_counts().to_dict())

    return CalibrationReport(
        n_chains=trusted.set_index(["neuron", "chain_idx"]).index.nunique(),
        n_chains_bad_anchor=len(bad_anchor),
        tp=tp, fp=fp, precision=precision,
        sampled_unflagged=n_smp, sampled_wrong=wrong_smp,
        silent_error_rate=silent_rate, est_recall=est_recall,
        by_error_type=by_type,
    )


def print_report(rep: CalibrationReport) -> None:
    print("=== QC calibration (trusted-anchor chains only) ===")
    print(f"chains scored: {rep.n_chains}  | excluded for bad anchor: {rep.n_chains_bad_anchor}")
    print(f"precision (flag => real error): {rep.precision:.2f}  (TP={rep.tp}, FP={rep.fp})")
    print(f"silent-error rate (wrong among un-flagged sample): {rep.silent_error_rate:.2%} "
          f"({rep.sampled_wrong}/{rep.sampled_unflagged})")
    print(f"estimated recall: {rep.est_recall:.2f}")
    print(f"error types among true errors: {rep.by_error_type}")
    print("\nReadout: low precision => thresholds too tight (tune them, e.g. "
          "skeleton_dilation_px); non-trivial silent-error rate => signals miss a "
          "failure mode (the un-flagged-but-wrong frames are the ones to study).")


# ---------------------------------------------------------------------------
# Stage C' — offline threshold sweep against the FIXED labels
# ---------------------------------------------------------------------------
def sweep_thresholds(
    labels_csv: str | Path,
    output_root: str | Path,
    *,
    skeleton,                       # annotate_df (or per-cell df) for compute_metrics
    grid: Optional[dict] = None,
    save_downscale: int = 4,
    scale: int = 8,
) -> pd.DataFrame:
    """Re-score saved masks under different qc thresholds and measure precision /
    silent-error rate against the human labels (labels are fixed; only the
    predicted flags move). No re-segmentation — qc.compute_metrics reads the
    masks off disk.

    grid example:
        {"skeleton_dilation_px": [3, 5, 7],
         "temporal_iou_min":     [0.2, 0.3],
         "area_ratio_bounds":    [(0.5, 2.0), (0.4, 2.5)]}
    """
    from itertools import product
    from sam2_utils import qc as qcmod

    grid = grid or {"skeleton_dilation_px": [3, 5, 7]}
    lab = pd.read_csv(labels_csv)
    anchor = lab[lab.role == "anchor"]
    bad_anchor = set(map(tuple, anchor[anchor.verdict == "wrong"][["neuron", "chain_idx"]].values))
    lab = lab[~lab.set_index(["neuron", "chain_idx"]).index.isin(bad_anchor)]
    truth = lab.set_index(["neuron", "chain_idx", "z"]).verdict.eq("wrong").to_dict()
    output_root = Path(output_root)
    chains = {(n, int(c)) for n, c in lab[["neuron", "chain_idx"]].values}

    keys = list(grid)
    out = []
    for combo in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        tp = fp = 0
        for (neuron, ci) in chains:
            md = _chain_dir(output_root, neuron, ci) / "masks"
            df = qcmod.compute_metrics(md, skeleton, cell_name=neuron,
                                       scale=scale, save_downscale=save_downscale,
                                       **params)
            for _, r in df.iterrows():
                key = (neuron, ci, int(r.z))
                if key not in truth:        # only frames we actually labelled
                    continue
                if bool(r.flag):
                    if truth[key]:
                        tp += 1
                    else:
                        fp += 1
        out.append({**params, "tp": tp, "fp": fp,
                    "precision": tp / (tp + fp) if (tp + fp) else float("nan")})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    print(__doc__)
    print("This module is a library + notebook workflow; it does not auto-run a "
          "labeling session. See the docstring 'Workflow' block.")