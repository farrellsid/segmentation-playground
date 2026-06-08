"""
labels.py — the M4 GUI's per-frame label store (the "label engine").

Every correction a human makes in the napari review GUI is a *training label*.
This module owns the on-disk store those labels accrue into — one flat per-frame
row — so the M4.5 predictor milestone can train a learned ``P(error)`` detector on
the exhaust of M4 review (PIPELINE_CONTEXT §7 *GUI as label engine*).

The split is deliberate and is the M4 / M4.5 boundary:
    M4 (here)   COLLECTS labels — the GUI appends rows as the human works.
    M4.5        TRAINS on them — logistic / small-GBT over the signal vector.
This module is the M4 half only. It does no modelling, no thresholding, no
re-segmentation; it is a pure, append-only CSV ledger with a fixed schema.

Why a dedicated store (and why it reuses calibration.py's schema)
-----------------------------------------------------------------
``calibration.py`` sketched a gold-set labeling schema but was shelved (manual
labeling was too much effort up front — PIPELINE_CONTEXT §7 *Manual gold-set
labeling*). Its two structural facts still bind M4, and this store enforces them:

  1. **Anchor contamination poisons labels.** A frame that is wrong because the
     *anchor* was wrong is not a propagation-signal failure; training on it teaches
     the model to predict anchor failures from features that can't see anchors.
     => every row carries the chain's anchor verdict (``anchor_passed`` /
        ``anchor_reasons`` / ``anchor_contained``) so M4.5 can exclude or separately
        model bad-anchor chains.
  2. **Selection bias is the killer.** Labels collected *only* on flagged frames are
     censored: a model trained on them can cut false positives but can never learn
     to catch what the rule misses (silent errors), because it never sees the
     stable-but-wrong regime. => the store records a ``role`` per row and provides
     ``sample_unflagged`` so the GUI can log a uniform random sample of *un-flagged*
     "good" frames too. Non-negotiable per §7; without it the eventual model only
     shrinks the queue, never widens coverage.

Schema (one flat row per labelled frame)
-----------------------------------------
identity      neuron, chain_idx, z
role          why this frame is in the store — see ROLES
features       the QC signal vector at label time (the model's inputs):
              flag_count, rule_flagged, area, n_components, skeleton_contained,
              area_ratio, temporal_iou, pred_iou, logit_conf
anchor verdict anchor_passed, anchor_reasons, anchor_contained  (chain-level,
              repeated on every row of the chain — the §7 anchor-contamination guard)
human label    verdict ('ok' | 'wrong'), error_type (one of ERROR_TYPES or '')
provenance     source (what GUI action produced the row), reviewer, ts, notes

Idempotent per (neuron, chain_idx, z): re-labelling a frame overwrites its prior
row, so a reviewer can revisit a frame without piling up duplicates. The store
joins to ``qc.csv`` and ``_manifest.csv`` on (neuron, chain_idx[, z]).

Torch-free / napari-free by design — same as ``qc`` / ``alignment`` — so the
label schema can be exercised and unit-tested on any box (see
``tests/test_labels.py``). The GUI imports it; it does not import the GUI.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


# Verdict vocabulary. 'ok' is the implicit good label; these tag *why* a mask is
# wrong, which is what makes the eventual detector diagnosable per failure mode.
# Mirrors calibration.ERROR_TYPES (kept in sync deliberately — same meaning) so a
# shelved-tool gold set and a GUI-collected set share one vocabulary.
ERROR_TYPES = ("wrong_object", "under", "over", "bleed", "fragmented", "missing", "other")

# Frame roles — why a row exists. Drives the train/eval split and the §7 selection-
# bias guard (sampled rows are the only window onto silent errors).
ROLES = ("anchor", "flagged", "sampled", "corrected")

# The flat per-frame schema. Order is the on-disk column order.
LABEL_COLS = [
    # identity
    "neuron", "chain_idx", "z",
    # role + whether the rule flagged it
    "role", "rule_flagged", "flag_count",
    # QC signal vector (the model features)
    "area", "n_components", "skeleton_contained",
    "area_ratio", "temporal_iou", "pred_iou", "logit_conf",
    # chain-level anchor verdict (anchor-contamination guard)
    "anchor_passed", "anchor_reasons", "anchor_contained",
    # human label
    "verdict", "error_type",
    # provenance
    "source", "reviewer", "ts", "notes",
]

# QC columns lifted verbatim from a qc.csv row into the feature block. Names match
# qc.compute_metrics output exactly, so record_from_qc is a straight copy.
_QC_FEATURE_COLS = [
    "flag_count", "area", "n_components", "skeleton_contained",
    "area_ratio", "temporal_iou", "pred_iou", "logit_conf",
]

# The (neuron, chain_idx, z) primary key — one row per labelled frame.
_KEY = ["neuron", "chain_idx", "z"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file in the same dir, then os.replace — a crash mid-write
    can't corrupt the store (same pattern as batch._atomic_write_csv). Important
    once the GUI and a background batch can both touch the output tree (§7
    parallel review)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _clean(v):
    """Coerce a value to something CSV/JSON-clean. numpy scalars -> python, NaN
    -> '' for the bool-ish tri-state columns handled by the caller; pass-through
    otherwise."""
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


class LabelStore:
    """Append-only per-frame label ledger backed by one CSV (default
    ``<output_root>/_labels.csv``).

    Usage (from the GUI)::

        store = LabelStore(output_root)
        # human approves a frame as correct:
        store.record(neuron, chain_idx, z, verdict="ok", role="flagged",
                     source="approve", reviewer="sf",
                     qc_row=qc_df.loc[z], anchor=state.anchor_score)
        # human marks a frame wrong and corrects it:
        store.record(neuron, chain_idx, z, verdict="wrong", error_type="bleed",
                     role="corrected", source="resume", reviewer="sf",
                     qc_row=qc_df.loc[z], anchor=state.anchor_score)
        # log the silent-error window (uniform un-flagged sample), all 'ok':
        store.sample_unflagged(neuron, chain_idx, qc_df, n=8, reviewer="sf")

    Every write is the full-file atomic rewrite — fine at label volumes (a human
    produces labels slowly; §7 "data volume is not the constraint, label coverage
    is").
    """

    def __init__(self, output_root: Union[str, Path], *, filename: str = "_labels.csv"):
        self.path = Path(output_root) / filename

    # -- read ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return the store as a DataFrame (empty with the right columns if new)."""
        if self.path.exists():
            df = pd.read_csv(self.path)
            return df.reindex(columns=LABEL_COLS)
        return pd.DataFrame(columns=LABEL_COLS)

    def __len__(self) -> int:
        return 0 if not self.path.exists() else len(pd.read_csv(self.path))

    # -- write -----------------------------------------------------------------
    def record(
        self,
        neuron: str,
        chain_idx: int,
        z: int,
        *,
        verdict: str,
        role: str = "flagged",
        error_type: str = "",
        source: str = "",
        reviewer: str = "",
        notes: str = "",
        qc_row: Optional[Mapping] = None,
        anchor: Optional[Mapping] = None,
        rule_flagged: Optional[bool] = None,
    ) -> dict:
        """Record (or overwrite) the label for one frame. Returns the row dict.

        Parameters
        ----------
        verdict : 'ok' | 'wrong'
            The human's call. 'ok' = mask is correct; 'wrong' = needs/needed a fix.
        role : one of ROLES
            'anchor'   — the seed frame; its verdict gates the whole chain.
            'flagged'  — the rule queued it (the precision set).
            'sampled'  — the rule called it clean; a random-sample row (the recall
                         / silent-error window — see sample_unflagged).
            'corrected'— the human edited/re-segmented this frame.
        error_type : one of ERROR_TYPES (only meaningful when verdict='wrong')
        qc_row : Mapping, optional
            A row of a qc.csv (e.g. ``qc_df.loc[z]``). Its QC signals are copied
            into the feature block verbatim. ``rule_flagged`` defaults to
            ``flag_count >= 1`` from this row unless overridden.
        anchor : Mapping, optional
            The chain's anchor verdict — ``ChainState.anchor_score`` (the dict from
            ``pipeline._anchor_score_to_dict``). Copied to anchor_* columns.
        rule_flagged : bool, optional
            Override the rule-flagged flag (else derived from qc_row.flag_count).
        """
        if verdict not in ("ok", "wrong"):
            raise ValueError(f"verdict must be 'ok' or 'wrong', got {verdict!r}")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if error_type and error_type not in ERROR_TYPES:
            raise ValueError(f"unknown error_type {error_type!r}; allowed: {ERROR_TYPES}")

        row = {c: "" for c in LABEL_COLS}
        row.update(neuron=neuron, chain_idx=int(chain_idx), z=int(z),
                   role=role, verdict=verdict, error_type=error_type,
                   source=source, reviewer=reviewer, notes=notes, ts=_now())

        # feature block from the qc row (if given)
        if qc_row is not None:
            for c in _QC_FEATURE_COLS:
                if c in qc_row:
                    row[c] = _clean(qc_row[c])
            fc = qc_row.get("flag_count", None) if hasattr(qc_row, "get") else None
            if rule_flagged is None and fc is not None and pd.notna(fc):
                rule_flagged = bool(int(fc) >= 1)
        row["rule_flagged"] = "" if rule_flagged is None else bool(rule_flagged)

        # anchor verdict block (chain-level, repeated per row)
        if anchor:
            row["anchor_passed"] = _clean(anchor.get("passed", ""))
            reasons = anchor.get("reasons", [])
            row["anchor_reasons"] = ",".join(reasons) if reasons else ""
            contained = anchor.get("contained", None)
            row["anchor_contained"] = "" if contained is None else bool(contained)

        self._append(row)
        return row

    def sample_unflagged(
        self,
        neuron: str,
        chain_idx: int,
        qc_df: pd.DataFrame,
        *,
        n: int = 8,
        seed: int = 0,
        reviewer: str = "",
        anchor: Optional[Mapping] = None,
        exclude_z: Sequence[int] = (),
    ) -> list[dict]:
        """Log a uniform random sample of *un-flagged* frames as role='sampled',
        verdict='ok'.

        This is the §7 selection-bias guard made concrete: the only window onto
        silent errors (stable-but-wrong masks the rule never flagged). The sample
        is **uniform** on purpose — a biased sample would bias the silent-error
        rate the M4.5 eval set estimates. The reviewer is expected to actually look
        at these frames and *downgrade* any that are wrong (call ``record(...,
        verdict='wrong', role='sampled')`` to overwrite). Logging them 'ok' up
        front means an un-reviewed sample is conservatively assumed correct.

        ``qc_df`` is a chain's qc.csv (z-indexed or with a ``z`` column).
        """
        df = qc_df.copy()
        if df.index.name != "z" and "z" in df.columns:
            df = df.set_index("z")
        # un-flagged = flag (>=1 signal) is False / absent
        flagcol = "flag" if "flag" in df.columns else None
        if flagcol is not None:
            unflagged = df.index[df[flagcol] != True].astype(int).tolist()   # noqa: E712
        else:
            unflagged = df.index.astype(int).tolist()
        excl = {int(z) for z in exclude_z}
        unflagged = [z for z in unflagged if z not in excl]

        k = min(int(n), len(unflagged))
        if k == 0:
            return []
        rng = np.random.default_rng(seed)
        chosen = sorted(int(z) for z in rng.choice(unflagged, size=k, replace=False))

        out = []
        for z in chosen:
            qc_row = df.loc[z] if z in df.index else None
            out.append(self.record(
                neuron, chain_idx, z, verdict="ok", role="sampled",
                source="sample", reviewer=reviewer, qc_row=qc_row, anchor=anchor))
        return out

    # -- internals -------------------------------------------------------------
    def _append(self, row: dict) -> None:
        """Upsert one row keyed on (neuron, chain_idx, z): drop any existing row
        with the same key, then append the new one. Full atomic rewrite."""
        df = self.load()
        new = pd.DataFrame([row], columns=LABEL_COLS)
        if len(df):
            key_match = ((df["neuron"] == row["neuron"])
                         & (df["chain_idx"] == row["chain_idx"])
                         & (df["z"] == row["z"]))
            df = df[~key_match]
        df = pd.concat([df, new], ignore_index=True)
        _atomic_write_csv(df, self.path)

    # -- diagnostics -----------------------------------------------------------
    def summary(self) -> dict:
        """Counts by role and verdict — a quick 'how much label coverage so far'."""
        df = self.load()
        if not len(df):
            return {"n": 0, "by_role": {}, "by_verdict": {}}
        return {
            "n": int(len(df)),
            "by_role": df["role"].value_counts().to_dict(),
            "by_verdict": df["verdict"].value_counts().to_dict(),
            "n_chains": int(df.set_index(_KEY[:2]).index.nunique()),
        }
