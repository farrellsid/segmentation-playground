"""
run_erl.py, compute per-neuron ERL on the cross-worm GT, end to end.

This wires the four pieces together (skeletons + registration + GT labelmaps +
the ERL metric) into a per-neuron ERL and a split/merge breakdown. Two modes:

* ``--mode self`` (default), **GT self-consistency check.** The "prediction" is
  the GT labelmap itself: sample it at each skeleton node *through the fitted
  registration*, map the sampled segment ``Nr`` to its GT neuron label, and run
  ERL. If the registration is good, each node lands on its own neuron's segment,
  so ERL should approach the perfect-segmentation **ceiling** (≈76 µm on p280).
  This validates the whole chain before any model prediction exists: the gap
  from the ceiling is exactly registration / GT-coverage error, expressed in µm.

* ``--mode pred``, score a real prediction. Same node sampling, but the sampled
  pixel value is used as the predicted object id directly (no Nr→neuron mapping),
  because predicted objects carry no neuron identity. Point ``--label-dir`` at a
  directory of per-slice predicted labelmaps on the GT grid (same ``s###`` naming
  as the GT export); falls back to the GT masks if omitted (== self mode).

Reads the GT via ``GroundTruth.from_config`` (now the stable F: drive). Run:

    py -3 -m eval.run_erl                       # self-consistency, all neurons
    py -3 -m eval.run_erl --neuron-limit 40     # quick subset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Hashable, Optional

import numpy as np

from sam2_utils.skeletons import normalize_name
from .erl import (Skeletons, expected_run_length, load_skeletons, per_neuron_erl,
                  sample_node_labels)
from .groundtruth import GroundTruth, _SLICE_IDX
from .registration import Registration


class _DirLabelmaps:
    """Per-slice predicted labelmaps in a directory: ``*_s###.png`` (uint16).

    The pred-mode label source for :func:`sample_node_labels`, same slice-index
    naming as the GT export, pixel value == predicted object id, 0 == background.
    This is the contract :mod:`eval.predict_gt` writes to (``labelmaps/``).
    """

    def __init__(self, root) -> None:
        from collections import OrderedDict
        self.root = Path(root)
        files = {}
        for p in self.root.glob("*.png"):
            m = _SLICE_IDX.search(p.name)
            if m:
                files[int(m.group(1))] = p
        self._files = OrderedDict(sorted(files.items()))

    def slice(self, z: int) -> np.ndarray:
        from PIL import Image
        if int(z) not in self._files:
            raise KeyError(z)
        arr = np.asarray(Image.open(self._files[int(z)]))
        return arr if arr.dtype == np.uint16 else arr.astype(np.uint16)


def _nr_to_label(gt: GroundTruth) -> Dict[int, str]:
    """Map GT segment ``Nr`` -> normalized neuron label (matches skel neuron ids).

    Uses the same :func:`normalize_name` the skeletons use, so a node landing on
    its own neuron's segment gets a label string equal to its ``node_neuron``.
    """
    md = gt.metadata
    return {int(nr): normalize_name(str(lab))
            for nr, lab in zip(md["nr"], md["label"]) if int(nr) != 0}


def _subset(skel: Skeletons, neurons: set) -> Skeletons:
    """Restrict a skeleton forest to a set of neuron ids (for quick runs)."""
    xyz = {n: p for n, p in skel.xyz.items() if skel.neuron.get(n) in neurons}
    neuron = {n: skel.neuron[n] for n in xyz}
    edges = [(u, v, L) for (u, v, L) in skel.edges if u in xyz and v in xyz]
    return Skeletons(edges=edges, xyz=xyz, neuron=neuron)


def run(
    skeleton_csv: Path,
    registration_json: Path,
    *,
    mode: str = "self",
    label_dir: Optional[Path] = None,
    neuron_limit: Optional[int] = None,
    merge_tol_frac: float = 0.0,
    merge_tol_count: int = 1,
) -> Dict[str, object]:
    """Compute the ERL report. Returns the summary dict (also has per-neuron)."""
    skel = load_skeletons(skeleton_csv)
    reg = Registration.from_json(registration_json)
    gt = GroundTruth.from_config()

    if neuron_limit:
        keep = set(sorted(set(skel.neuron.values()))[:neuron_limit])
        skel = _subset(skel, keep)

    # the label source: GT masks (self) or a predicted-labelmap dir (pred)
    if mode == "pred":
        if label_dir is None:
            raise ValueError("--mode pred requires --label-dir: predicted labelmaps "
                             "on the GT grid (*_s###.png, uint16, 0=bg). See predict_gt.py.")
        label_slice_fn: Callable[[int], np.ndarray] = _DirLabelmaps(label_dir).slice
    else:
        label_slice_fn = gt.label_slice

    # sample the label source at each node through the registration transform
    sampled = sample_node_labels(skel, label_slice_fn, transform=reg.transform)

    # build node_label:
    #   self, map sampled GT Nr -> GT neuron label, so a neuron split into
    #           several GT fragments still reads as one object along its skeleton.
    #   pred, use the predicted object id directly (predictions carry no neuron
    #           identity; a merge is one predicted id spanning >1 skeleton neuron).
    nr2lab = _nr_to_label(gt)
    node_label: Dict[str, Hashable] = {}
    on_seg = 0
    for n, raw in sampled.items():
        if mode == "pred":
            lab = int(raw) if raw else 0
        else:
            lab = nr2lab.get(int(raw), 0) if raw else 0    # 0 == background
        node_label[n] = lab
        if mode == "self" and lab and lab == skel.neuron.get(n):
            on_seg += 1

    report = expected_run_length(skel.edges, node_label, skel.neuron,
                                 min_support_frac=merge_tol_frac,
                                 min_support_count=merge_tol_count)
    per_neuron = per_neuron_erl(skel.edges, node_label, skel.neuron,
                                min_support_frac=merge_tol_frac,
                                min_support_count=merge_tol_count)

    # ceiling for this (possibly subset) skeleton set
    ceil = expected_run_length(skel.edges,
                               {n: neu for n, neu in skel.neuron.items()},
                               skel.neuron)

    report["erl_um"] = report["erl"] / 1000.0
    report["ceiling_um"] = ceil["erl"] / 1000.0
    report["pct_of_ceiling"] = (report["erl"] / ceil["erl"] * 100.0) if ceil["erl"] else 0.0
    report["total_length_um"] = report["total_length"] / 1000.0
    report["n_nodes_sampled"] = len(sampled)
    report["n_nodes_on_own_segment"] = on_seg
    report["on_segment_rate"] = (on_seg / len(sampled)
                                 if (mode == "self" and sampled) else float("nan"))
    report["_per_neuron"] = per_neuron
    return report


def _write_outputs(report: Dict[str, object], out_json: Path) -> None:
    per_neuron = report.pop("_per_neuron")
    out_json.write_text(json.dumps(report, indent=2))
    csv_path = out_json.with_suffix(".per_neuron.csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["neuron", "erl_um", "total_length_um", "n_runs", "max_run_um",
                    "n_split_edges", "n_bg_edges", "n_merge_edges"])
        for neu in sorted(per_neuron, key=lambda k: per_neuron[k]["erl"], reverse=True):
            d = per_neuron[neu]
            w.writerow([neu, round(d["erl"] / 1000, 3), round(d["total_length"] / 1000, 3),
                        d["n_runs"], round(d["max_run"] / 1000, 3),
                        d["n_split_edges"], d["n_bg_edges"], d["n_merge_edges"]])
    print(f"[erl] wrote {out_json}  and  {csv_path}")


def _main() -> None:
    base = Path(__file__).resolve().parent.parent / "data" / "groundtruth" / "skeletons_p280"
    ap = argparse.ArgumentParser(description="Per-neuron ERL on the cross-worm GT.")
    ap.add_argument("--skeleton-csv", type=Path, default=base / "aggregate_data_pv.csv")
    ap.add_argument("--registration", type=Path, default=base / "registration.json")
    ap.add_argument("--out", type=Path, default=base / "erl_self_consistency.json")
    ap.add_argument("--mode", choices=("self", "pred"), default="self")
    ap.add_argument("--label-dir", type=Path, default=None)
    ap.add_argument("--neuron-limit", type=int, default=None)
    ap.add_argument("--merge-tol-frac", type=float, default=0.0,
                    help="a neuron counts toward a label's merge only if it holds >= this "
                         "fraction of the label's nodes (0.0 = strict: any 2 neurons => merge)")
    ap.add_argument("--merge-tol-count", type=int, default=1,
                    help="...and >= this many nodes (default 1 = strict)")
    args = ap.parse_args()

    report = run(args.skeleton_csv, args.registration, mode=args.mode,
                 label_dir=args.label_dir, neuron_limit=args.neuron_limit,
                 merge_tol_frac=args.merge_tol_frac, merge_tol_count=args.merge_tol_count)
    print(f"[erl] mode={args.mode}  neurons scored: {len(report['_per_neuron'])}")
    print(f"[erl] on-segment rate: {report['on_segment_rate']:.1%} "
          f"({report['n_nodes_on_own_segment']}/{report['n_nodes_sampled']} nodes)")
    print(f"[erl] ERL = {report['erl_um']:.1f} µm   ceiling = {report['ceiling_um']:.1f} µm   "
          f"({report['pct_of_ceiling']:.0f}% of ceiling)")
    print(f"[erl] merges: {report['n_merge_labels']}   split-edges: {report['n_split_edges']}   "
          f"bg-edges: {report['n_bg_edges']}")
    _write_outputs(report, args.out)


if __name__ == "__main__":
    _main()
