"""
review_queue.py: the review GUI's work queue + review-status ledger.

The batch runner (``batch.py``) owns the *execution* status of every chain
(``_manifest.csv``: pending -> running -> done/flagged/failed) and rolls flagged
frames into ``_triage.csv``. The review GUI is a separate consumer: it needs to
know which *chains* still need a human and to record what the human decided,
**without** clobbering the execution status the batch owns.

So, for parallel review, the GUI owns
a **separate** ledger, ``_review.csv``, with its own status column:

    background batch  owns  _manifest.csv  (execution: pending->running->done/flagged)
    review GUI        owns  _review.csv    (review:   unreviewed->in_review->
                                            approved/rejected/corrected)

Keeping them in two files is the cheap version of partitioning ownership:
the two processes never write the same column, so the only thing a
file lock would add is protection against two writers of the *same* ledger. We
take the same atomic-rewrite tactic as ``batch._atomic_write_csv`` (temp file +
``os.replace``); a cross-process file lock (``filelock``/``portalocker``) is the
next step if a second GUI ever runs concurrently, see the not-implemented notes below.

Queue definition
----------------
A chain needs review when the batch flagged it, i.e. its manifest ``status`` is
``flagged`` (which, at the canonical ``qc_triage_min_signals=2``, means it has
>=1 intervene-level frame). The queue is those chains, minus any the human has
already dispositioned in ``_review.csv``. ``_triage.csv`` gives the per-frame
detail within a chain (which z's, why) and is read by the GUI when a chain opens;
this module works at chain granularity.

Not implemented this pass (parallel review)
--------------------------------------------------------------------------
  * **Cross-process file lock** around ``_review.csv`` writes. Single-reviewer is
    safe as-is (one writer); a lock is required only for concurrent GUIs. Marked
    at the write site.
  * **Live polling / fs-watch** so chains flagged by a still-running batch appear
    mid-session. ``refresh()`` re-reads from disk on demand (poll-on-demand); a
    timer/watchdog auto-poll is a GUI-loop concern left to gui.py.
  * **GPU arbitration** for interactive re-runs vs. background batch: an
    infra/runtime concern, not a ledger concern; lives in gui.py / a future
    multi-GPU harness, not here.

Torch-free / napari-free, like ``labels`` and ``qc``; unit-tested in
``tests/test_review_queue.py``.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Review-status vocabulary (GUI-owned; distinct from manifest execution status).
UNREVIEWED = "unreviewed"   # flagged by the batch, no human disposition yet
IN_REVIEW = "in_review"     # a reviewer has it open (claim); a crashed session is retried
APPROVED = "approved"       # human confirmed the auto masks are fine as-is
REJECTED = "rejected"       # human judged it unfixable / to be redone (e.g. bad anchor)
CORRECTED = "corrected"     # human edited prompts/masks and re-saved
REVIEW_STATUSES = (UNREVIEWED, IN_REVIEW, APPROVED, REJECTED, CORRECTED)
# Terminal dispositions: a chain in one of these drops out of the pending queue.
DONE_STATUSES = {APPROVED, REJECTED, CORRECTED}

REVIEW_COLUMNS = ["neuron", "chain_idx", "review_status", "reviewer", "notes", "updated_at"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Temp file + os.replace (mirrors batch._atomic_write_csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


class ReviewQueue:
    """Chains-needing-review queue + the GUI-owned ``_review.csv`` disposition ledger.

    Read paths (from the batch's artifacts, never mutated here):
        <output_root>/_manifest.csv   # execution status; 'flagged' => needs review
        <output_root>/_triage.csv     # per-frame detail (read by the GUI per chain)
    Write path (owned here):
        <output_root>/_review.csv     # review_status per (neuron, chain_idx)

    Typical GUI loop::

        q = ReviewQueue(output_root)
        for neuron, chain_idx in q.pending():     # flagged & not yet dispositioned
            q.claim(neuron, chain_idx, reviewer="sf")     # -> in_review
            ... open the chain, let the human work ...
            q.set_status(neuron, chain_idx, CORRECTED, reviewer="sf")
        q.refresh()   # re-read manifest to pick up chains a running batch flagged
    """

    def __init__(self, output_root: Union[str, Path]):
        self.output_root = Path(output_root)
        self.manifest_path = self.output_root / "_manifest.csv"
        self.triage_path = self.output_root / "_triage.csv"
        self.review_path = self.output_root / "_review.csv"
        self._manifest: Optional[pd.DataFrame] = None

    # -- manifest (read-only here) ---------------------------------------------
    def refresh(self) -> pd.DataFrame:
        """(Re)read _manifest.csv from disk. Call to pick up chains a still-running
        batch has flagged since the GUI started (auto-poll is not implemented: a timer
        in the GUI loop would call this; here it's poll-on-demand)."""
        if not self.manifest_path.exists():
            self._manifest = pd.DataFrame(columns=["neuron", "chain_idx", "status"])
        else:
            self._manifest = pd.read_csv(self.manifest_path)
        return self._manifest

    @property
    def manifest(self) -> pd.DataFrame:
        if self._manifest is None:
            self.refresh()
        return self._manifest

    def flagged_chains(self) -> List[tuple]:
        """All chains the batch marked 'flagged', as (neuron, chain_idx), in
        manifest order."""
        m = self.manifest
        if "status" not in m.columns:
            return []
        sel = m[m["status"] == "flagged"]
        return [(str(r["neuron"]), int(r["chain_idx"])) for _, r in sel.iterrows()]

    # -- review ledger (owned here) --------------------------------------------
    def load_review(self) -> pd.DataFrame:
        if self.review_path.exists():
            return pd.read_csv(self.review_path).reindex(columns=REVIEW_COLUMNS)
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    def status_of(self, neuron: str, chain_idx: int) -> str:
        """Review status of a chain (UNREVIEWED if it has no ledger row yet)."""
        df = self.load_review()
        m = (df["neuron"] == neuron) & (df["chain_idx"] == int(chain_idx))
        return str(df.loc[m, "review_status"].iloc[0]) if m.any() else UNREVIEWED

    def set_status(self, neuron: str, chain_idx: int, status: str, *,
                   reviewer: str = "", notes: str = "") -> None:
        """Upsert a chain's review disposition. Idempotent per (neuron, chain_idx)."""
        if status not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of {REVIEW_STATUSES}, got {status!r}")
        df = self.load_review()
        row = {"neuron": neuron, "chain_idx": int(chain_idx), "review_status": status,
               "reviewer": reviewer, "notes": notes, "updated_at": _now()}
        if len(df):
            m = (df["neuron"] == neuron) & (df["chain_idx"] == int(chain_idx))
            df = df[~m]
        df = pd.concat([df, pd.DataFrame([row], columns=REVIEW_COLUMNS)], ignore_index=True)
        _atomic_write_csv(df, self.review_path)   # not implemented: a file lock for concurrent GUIs

    def claim(self, neuron: str, chain_idx: int, *, reviewer: str = "") -> None:
        """Mark a chain in_review (a reviewer opened it). A session that crashes
        leaves it in_review; ``pending(include_in_review=True)`` re-surfaces it."""
        self.set_status(neuron, chain_idx, IN_REVIEW, reviewer=reviewer)

    # -- the queue ------------------------------------------------------------
    def pending(self, *, include_in_review: bool = True) -> List[tuple]:
        """Chains that need a human: flagged by the batch and not yet dispositioned.

        ``include_in_review`` (default True) re-surfaces chains left ``in_review``
        by a crashed/abandoned session, so work is never silently lost. Set False
        to hide chains another reviewer currently has open.
        """
        review = self.load_review()
        disposed = set()
        in_review = set()
        for _, r in review.iterrows():
            key = (str(r["neuron"]), int(r["chain_idx"]))
            st = str(r["review_status"])
            if st in DONE_STATUSES:
                disposed.add(key)
            elif st == IN_REVIEW:
                in_review.add(key)
        out = []
        for key in self.flagged_chains():
            if key in disposed:
                continue
            if key in in_review and not include_in_review:
                continue
            out.append(key)
        return out

    def triage_for(self, neuron: str, chain_idx: int) -> pd.DataFrame:
        """The per-frame triage rows (z + reasons + signals) for one chain, read
        from _triage.csv. Empty frame if absent. The GUI uses this to jump the
        reviewer straight to the queued frames within a chain."""
        if not self.triage_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.triage_path)
        if not {"neuron", "chain_idx"}.issubset(df.columns):
            return pd.DataFrame()
        m = (df["neuron"] == neuron) & (df["chain_idx"] == int(chain_idx))
        return df[m].copy()

    def summary(self) -> dict:
        """Queue health: how many flagged, how many dispositioned, what's left."""
        review = self.load_review()
        by_status = (review["review_status"].value_counts().to_dict()
                     if len(review) else {})
        flagged = self.flagged_chains()
        return {
            "n_flagged": len(flagged),
            "n_pending": len(self.pending()),
            "by_review_status": by_status,
        }
