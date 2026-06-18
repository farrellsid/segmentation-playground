"""QC over a finished chain's saved masks: compute metrics, write qc.csv, set the verdict."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from sam2_utils import alignment

    from .config import PipelineConfig


def run_qc(masks_dir: Path, skeleton: pd.DataFrame, *,
           frame_to_z: dict[int, int],
           frame_conf: Optional[dict[int, float]],
           pred_iou: Optional[dict[int, float]] = None,
           cfg: "PipelineConfig",
           qc_csv_path: Optional[Path] = None,
           crop_window: Optional["alignment.CropWindow"] = None,
           ) -> tuple[dict, list[int], str]:
    """Compute QC over the saved masks, write qc.csv, return (summary, triage_z, status).

    QC runs over the
    just-saved chain (joining the inline-captured confidence), produces flags, and
    drives the chain's verdict -> all headless, no human. It still reads the PNGs
    back off disk rather than scoring inside the propagate loop; that fully-inline,
    interleaved form is only required for *halt-and-re-prompt*, which is not built
    yet. So this is "QC moved into the run," not yet "QC moved into the propagation loop."

    Signals and the composite flag/intervene rule come straight from
    ``qc.compute_metrics`` (single source of truth); thresholds come from ``cfg``.

    Parameters
    ----------
    skeleton : DataFrame
        The skeleton of *this chain only* (columns z, x_tif, y_tif), NOT the whole
        neuron. This matters: a neuron like AVAL is many chains, so its nodes cross
        a given z at several xy positions and their centroid lands off any single
        process -> using it makes containment fail on every frame (the AVAL 100%-flag
        bug). Filtering to the chain's own nodes gives a meaningful per-z probe.

    Returns
    -------
    qc_summary : dict (json-safe)   -> counts + worst frames, for ChainState
    triage_z   : list[int]          -> CATMAID-z of every flagged frame (the queue;
                                      z-keyed to match qc, mask filenames, and
                                      review.load_chain's triage_is_z default)
    status     : "done" | "flagged"
    """
    from sam2_utils import qc   # lazy: keeps pipeline import free of qc's heavy deps

    # Invariant: pipeline.save_masks writes masks at _sam (scale) and never
    # resamples, so the on-disk mask space IS `scale`. qc.compute_metrics divides
    # the _tif skeleton by `save_downscale` to land in mask space, so the two must
    # be equal or QC silently mis-locates every node. The
    # canonical rule already enforces this; the guard turns a future divergence
    # from a silent wrong-QC run into a loud failure. If you ever want resampled,
    # higher-res Blender masks, make save_masks resample first, then relax this.
    # The scale==save_downscale guard protects the _sam node lookup (skeleton / scale).
    # Tier-2 masks live in _pcrop, where the node lookup goes through crop_window
    # instead of / save_downscale, so the guard does not apply, skip it when a
    # crop_window is supplied (and the node mapping is overridden below).
    if crop_window is None and cfg.scale != cfg.save_downscale:
        raise ValueError(
            f"run_qc: scale ({cfg.scale}) != save_downscale ({cfg.save_downscale}), "
            "but pipeline.save_masks does not resample, so the on-disk masks are at "
            "`scale`. QC would divide the skeleton by save_downscale and mis-locate "
            "every node. Keep save_downscale == scale (canonical), or make save_masks "
            "resample to save_downscale before changing this."
        )

    # pred_iou comes in frame_idx-keyed (from PropagationSession); compute_metrics is
    # z-indexed, so remap. Once joined, the `pred_iou` column becomes the 4th flag-rule
    # signal (cfg.qc_pred_iou_min); it was inert while NaN. See propagate()'s note re:
    # clearing the manifest after enabling it.
    pred_iou_z = None
    if pred_iou:
        pred_iou_z = {frame_to_z[fi]: v for fi, v in pred_iou.items()
                      if fi in frame_to_z}

    # In _pcrop the node-containment radius is rescaled by scale/crop_scale (same
    # space_ratio run_chain applies to the anchor gate), so the physical tolerance
    # matches the _sam path; compute_metrics maps nodes _tif->_pcrop via crop_window.
    dilation_px = cfg.qc_skeleton_dilation_px
    if crop_window is not None:
        dilation_px = int(round(cfg.qc_skeleton_dilation_px
                                * crop_window.sam_scale / crop_window.crop_scale))

    df = qc.compute_metrics(
        masks_dir,
        skeleton=skeleton,
        scale=cfg.scale,
        save_downscale=cfg.save_downscale,
        pred_iou=pred_iou_z,
        skeleton_dilation_px=dilation_px,
        area_ratio_bounds=cfg.qc_area_ratio_bounds,
        temporal_iou_min=cfg.qc_temporal_iou_min,
        pred_iou_min=cfg.qc_pred_iou_min,
        crop_window=crop_window,
    )

    # Attach the inline confidence proxy as a *diagnostic* column (z-keyed).
    # Deliberately NOT named pred_iou and NOT in the flag rule -> see propagate().
    if frame_conf:
        z_conf = {frame_to_z[fi]: c for fi, c in frame_conf.items()
                  if fi in frame_to_z}
        df["logit_conf"] = df.index.map(lambda z: z_conf.get(int(z), float("nan")))

    # The human triage queue is the frames at/above the configured severity
    # (flag_count >= qc_triage_min_signals; default 2 = intervene-level). `flag`
    # (>=1 signal) stays in the row as a diagnostic: single-signal flags are kept on
    # disk for labels, just not surfaced to a human. Persisted to qc.csv so the
    # cross-chain rollup (batch.build_triage_queue) can filter on the artifact alone.
    df["queue"] = df["flag_count"] >= cfg.qc_triage_min_signals

    if qc_csv_path is not None:
        qc_csv_path = Path(qc_csv_path)
        qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(qc_csv_path)   # index = z

    n = int(len(df))
    n_flag = int(df["flag"].sum())
    n_int = int(df["intervene"].sum())
    n_queue = int(df["queue"].sum())
    n_noskel = int((df["skeleton_contained"] == False).sum())   # noqa: E712
    n_skel_na = int(df["skeleton_contained"].isna().sum())
    triage_z = sorted(int(z) for z in df.index[df["queue"]])

    # worst queue frames first, for a quick human glance from state.json alone
    worst = (df[df["queue"]].sort_values("flag_count", ascending=False).head(10))
    worst_frames = [
        {
            "z": int(z),
            "flag_count": int(r["flag_count"]),
            "area_ratio": (None if pd.isna(r["area_ratio"]) else round(float(r["area_ratio"]), 3)),
            "temporal_iou": (None if pd.isna(r["temporal_iou"]) else round(float(r["temporal_iou"]), 3)),
            "skeleton_contained": bool(r["skeleton_contained"]),
        }
        for z, r in worst.iterrows()
    ]

    qc_summary = {
        "n_frames": n,
        "n_flagged": n_flag,          # all >=1-signal flags (diagnostic; kept for labels)
        "n_queue": n_queue,           # frames surfaced to a human (the triage gate)
        "n_intervene": n_int,
        "n_missing_skel": n_noskel,
        "n_skel_not_assessable": n_skel_na,
        "flag_rate": (round(n_flag / n, 4) if n else 0.0),
        "queue_rate": (round(n_queue / n, 4) if n else 0.0),
        "thresholds": {
            "area_ratio_bounds": list(cfg.qc_area_ratio_bounds),
            "temporal_iou_min": cfg.qc_temporal_iou_min,
            "pred_iou_min": cfg.qc_pred_iou_min,
            "skeleton_dilation_px": cfg.qc_skeleton_dilation_px,
            "triage_min_signals": cfg.qc_triage_min_signals,
        },
        "worst_frames": worst_frames,
    }

    # chain verdict keyed on the SAME queue definition as the frame queue, so the two
    # never disagree. Behaviour-preserving at defaults: qc_triage_min_signals=2 makes
    # n_queue == n_intervene, so this is identical to the prior
    # `n_int >= qc_intervene_to_flag_chain` rule.
    status = "flagged" if n_queue >= cfg.qc_intervene_to_flag_chain else "done"
    return qc_summary, triage_z, status
