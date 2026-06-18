"""
erl.py, Expected Run Length (skeleton-based).

ERL is the *skeleton half* of the eval ruler: the **expected error-free traced
path length** measured from a
uniformly-random point along the ground-truth skeletons, with any predicted
segment that **merges two neurons** contributing zero length. The advance gate is
stated in it ("produce a per-neuron ERL and a
split/merge breakdown"). Source: Januszewski et al., Nature Methods 2018
(s41592-018-0049-4); the merge-zeroing is what makes it punish the costly error.

Why this is buildable now (no pixel data)
------------------------------------------
ERL needs two things: the skeletons (which true point connects to which) and a
**predicted label per skeleton node** (which predicted object each point fell in).
The skeletons are local (``data/groundtruth/skeletons_p280/``). Only the per-node
labels come from a real prediction, sampling the predicted labelmap at each
registered node, and that is the single drive-dependent wire-in, isolated in
:func:`sample_node_labels` (which takes a generic ``label_slice_fn`` so it tests
on synthetic arrays). The metric core below is pure: it takes an edge list + two
node→id maps and returns plain floats, so it unit-tests on toy skeletons with
hand-computed answers, exactly like :mod:`eval.metrics`.

The run-length math
-------------------
Let the skeletons be a forest of edges, each with a physical length. Give every
node a *true neuron* id (which arbor it traces) and a *predicted label* (which
segment the model put it in).

* A predicted label is a **merge** if nodes carrying it belong to >1 true neuron.
* An edge is **error-free** iff its two endpoints share a predicted label, that
  label is not background, and it is not a merge label.
* **Runs** are the connected components of the error-free edges; a run's length is
  the sum of its error-free edge lengths. (Merge labels form no runs → length 0.)
* For a uniformly-random point, E[run length] = Σ_runs len_run² / Σ_all len, so ERL = (Σ run_len²) / (total skeleton length). Splits shrink runs; merges and
  background drop length from the numerator while it stays in the denominator.

Pure / numpy + (pandas only in the loader). Coordinates in the skeleton CSV are
CATMAID **stack-px** (nm ÷ ``config.STACK_RESOLUTION_NM``); lengths here are
converted back to physical nm and reported in µm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple, Union

import numpy as np

try:                                   # importable without the package installed
    from sam2_utils import config
    from sam2_utils.skeletons import normalize_name
except Exception:                      # pragma: no cover - fallback for odd CWDs
    config = None

    def normalize_name(raw: str) -> str:  # minimal mirror of the real one
        s = str(raw).strip()
        if s.startswith("[") and "]" in s:
            s = s[1: s.rindex("]")]
        return s.rstrip("!?").strip()


_DEFAULT_RES_NM = (2.0, 2.0, 50.0)     # config.STACK_RESOLUTION_NM fallback

# Labels that mean "no predicted object here", these never form runs and never
# count toward a merge. ``None`` and 0 (the labelmap background) and "" are it.
_DEFAULT_BACKGROUND = (None, 0, "")


# =============================================================================
# Skeleton loading  (pandas; the only IO in this module)
# =============================================================================

def _canon_id(v) -> str:
    """Canonicalize a CATMAID node id to a single string form.

    The pulled CSV writes real (integer) node ids cleanly in ``node_id``
    (``"19412366"``) but float-formatted in ``parent_id`` (``"19424215.0"``),
    while virtual nodes are non-numeric (``"v_19408905_849"``). Without
    canonicalizing, every child→real-parent edge fails the join and the neuron
    trees shatter. Strip a trailing integral ``.0`` from numeric ids; leave
    ``v_*`` (and anything non-numeric) untouched.
    """
    s = str(v)
    if "." in s:
        head, _, tail = s.partition(".")
        if head.isdigit() and set(tail) <= {"0"}:
            return head
    return s

@dataclass
class Skeletons:
    """A skeleton forest ready for ERL: edges + node positions + node→neuron.

    ``edges`` are ``(child_id, parent_id, length_nm)``; ids are strings (CATMAID
    node ids, incl. ``v_*`` virtual nodes). ``xyz`` maps node id → stack-px
    ``(x, y, z)``; ``neuron`` maps node id → normalized neuron name.
    """
    edges: List[Tuple[str, str, float]]
    xyz: Dict[str, Tuple[float, float, float]]
    neuron: Dict[str, str]

    @property
    def total_length_nm(self) -> float:
        return float(sum(e[2] for e in self.edges))

    @property
    def neurons(self) -> List[str]:
        return sorted(set(self.neuron.values()))


def load_skeletons(
    csv_path: Union[str, Path],
    *,
    resolution_nm: Tuple[float, float, float] = None,
    normalize: Callable[[str], str] = normalize_name,
) -> Skeletons:
    """Read ``aggregate_data_pv.csv`` into a :class:`Skeletons` forest.

    One edge per non-root node (``parent_id`` not null), length = Euclidean
    distance between node and parent in **physical nm** (stack-px × resolution).
    Rows whose parent id is missing from the table are skipped (defensive; the
    pulled table is internally consistent). Neuron id = ``normalize(cell_name)``.
    """
    import pandas as pd
    if resolution_nm is None:
        resolution_nm = tuple(getattr(config, "STACK_RESOLUTION_NM", _DEFAULT_RES_NM))
    rx, ry, rz = (float(v) for v in resolution_nm)

    df = pd.read_csv(csv_path, dtype={"node_id": str, "parent_id": str})
    df = df.dropna(subset=["x", "y", "z"])
    xyz: Dict[str, Tuple[float, float, float]] = {
        _canon_id(nid): (float(x), float(y), float(z))
        for nid, x, y, z in zip(df["node_id"], df["x"], df["y"], df["z"])
    }
    neuron: Dict[str, str] = {
        _canon_id(nid): normalize(name)
        for nid, name in zip(df["node_id"], df["cell_name"].astype(str))
    }

    edges: List[Tuple[str, str, float]] = []
    for nid, pid in zip(df["node_id"], df["parent_id"]):
        if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pid in ("", "nan"):
            continue                         # root
        nid, pid = _canon_id(nid), _canon_id(pid)
        if pid not in xyz:
            continue                         # dangling parent (defensive)
        x1, y1, z1 = xyz[nid]
        x2, y2, z2 = xyz[pid]
        length = float(np.sqrt(((x1 - x2) * rx) ** 2
                               + ((y1 - y2) * ry) ** 2
                               + ((z1 - z2) * rz) ** 2))
        edges.append((nid, pid, length))
    return Skeletons(edges=edges, xyz=xyz, neuron=neuron)


# =============================================================================
# ERL core  (pure: edges + node→label + node→neuron -> floats)
# =============================================================================

class _DSU:
    """Tiny union-find over hashable node ids, tracking per-component length."""
    __slots__ = ("parent", "rank", "length")

    def __init__(self) -> None:
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}
        self.length: Dict[Hashable, float] = {}

    def find(self, x: Hashable) -> Hashable:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.length[x] = 0.0
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:        # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.length[ra] += self.length[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def add_length(self, x: Hashable, dl: float) -> None:
        self.length[self.find(x)] += dl

    def component_lengths(self) -> Dict[Hashable, float]:
        out: Dict[Hashable, float] = {}
        for x in self.parent:
            r = self.find(x)
            out[r] = self.length[r]
        return out


def _is_bg(label: Hashable, background: frozenset) -> bool:
    return label in background


def merge_labels(
    edges: Iterable[Tuple[str, str, float]],
    node_label: Dict[str, Hashable],
    node_neuron: Dict[str, str],
    *,
    background: Iterable = _DEFAULT_BACKGROUND,
    min_support_frac: float = 0.0,
    min_support_count: int = 1,
) -> Dict[Hashable, set]:
    """Predicted labels that span >1 true neuron -> ``{label: {contributor neurons}}``.

    Considers only nodes that actually appear on an edge (isolated nodes can't be
    traced), and ignores background labels.

    **Merge tolerance.** A neuron counts as a *contributor* to a label only if it
    carries at least ``min_support_count`` of that label's edge-nodes **and** at
    least ``min_support_frac`` of them. A label is a merge iff it has ≥2
    contributors. This stops a handful of stray nodes, e.g. a neighbour's
    skeleton drifting onto this segment through registration error, from flagging
    a whole neuron as a merge and zeroing its run length. The defaults
    (``count=1, frac=0.0``) reproduce the strict "any two neurons ⇒ merge" rule, so
    existing behaviour is unchanged unless a tolerance is passed.

    Caveat: tolerance only changes the merge *flag*. A minority node is not
    relabelled, so two adjacent minority nodes sharing a (tolerated) label can still
    form a short run attributed to their own neuron, bounded, and rare at sane
    thresholds. Reassigning minority nodes to background would remove even that; left
    out for now as a deliberate simplification.
    """
    bg = frozenset(background)
    counts: Dict[Hashable, Dict[str, int]] = {}
    totals: Dict[Hashable, int] = {}
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u); nodes.add(v)
    for n in nodes:
        lab = node_label.get(n)
        if _is_bg(lab, bg):
            continue
        neu = node_neuron.get(n)
        if neu is None:
            continue
        counts.setdefault(lab, {})
        counts[lab][neu] = counts[lab].get(neu, 0) + 1
        totals[lab] = totals.get(lab, 0) + 1
    out: Dict[Hashable, set] = {}
    for lab, per_neuron in counts.items():
        tot = totals[lab]
        thresh = max(int(min_support_count), min_support_frac * tot)
        contributors = {neu for neu, c in per_neuron.items() if c >= thresh}
        if len(contributors) > 1:
            out[lab] = contributors
    return out


def expected_run_length(
    edges: Iterable[Tuple[str, str, float]],
    node_label: Dict[str, Hashable],
    node_neuron: Dict[str, str],
    *,
    background: Iterable = _DEFAULT_BACKGROUND,
    min_support_frac: float = 0.0,
    min_support_count: int = 1,
) -> Dict[str, object]:
    """Compute ERL + a split/merge breakdown over a skeleton forest.

    Parameters
    ----------
    edges : iterable of ``(u, v, length)``, undirected skeleton edges, length in
        any consistent unit (nm here). Both endpoints should have a neuron id.
    node_label : node id -> predicted segment label (any hashable; ``None``/0/""
        treated as background, no object).
    node_neuron : node id -> true neuron id (which arbor the node traces).
    background : label values that mean "no predicted object".

    Returns
    -------
    dict with
      ``erl``, Σ run_len² / total_length, same unit as ``length``.
      ``total_length``, Σ of all edge lengths (the denominator).
      ``n_runs``, number of non-empty error-free runs.
      ``max_run``, longest run length.
      ``n_merge_labels``, predicted labels merging ≥2 neurons.
      ``merge_detail``, ``{label: sorted([neurons...])}``.
      ``n_split_edges``, same-neuron edges broken by a label change
                              (both endpoints non-bg, neither a merge label).
      ``n_bg_edges``, edges with a background endpoint.
      ``n_merge_edges``, edges dropped because an endpoint is a merge label.
    """
    edges = list(edges)
    bg = frozenset(background)
    merges = merge_labels(edges, node_label, node_neuron, background=bg,
                          min_support_frac=min_support_frac,
                          min_support_count=min_support_count)
    merge_set = set(merges)

    dsu = _DSU()
    total = 0.0
    n_split = n_bg = n_merge_edges = 0
    for u, v, length in edges:
        total += length
        lu, lv = node_label.get(u), node_label.get(v)
        if _is_bg(lu, bg) or _is_bg(lv, bg):
            n_bg += 1
            continue
        if lu in merge_set or lv in merge_set:
            n_merge_edges += 1
            continue
        if lu != lv:                         # same arbor, different predicted label
            n_split += 1
            continue
        # error-free edge: union its endpoints, then add its length once to the
        # (now common) component, find() auto-inserts both nodes.
        dsu.find(u); dsu.find(v)
        dsu.union(u, v)
        dsu.add_length(u, length)

    comp = dsu.component_lengths()
    run_lengths = [L for L in comp.values() if L > 0]
    sum_sq = float(sum(L * L for L in run_lengths))
    erl = (sum_sq / total) if total > 0 else 0.0

    return {
        "erl": float(erl),
        "total_length": float(total),
        "n_runs": len(run_lengths),
        "max_run": float(max(run_lengths)) if run_lengths else 0.0,
        "n_merge_labels": len(merges),
        "merge_detail": {lab: sorted(neus) for lab, neus in merges.items()},
        "n_split_edges": int(n_split),
        "n_bg_edges": int(n_bg),
        "n_merge_edges": int(n_merge_edges),
    }


def per_neuron_erl(
    edges: Iterable[Tuple[str, str, float]],
    node_label: Dict[str, Hashable],
    node_neuron: Dict[str, str],
    *,
    background: Iterable = _DEFAULT_BACKGROUND,
    min_support_frac: float = 0.0,
    min_support_count: int = 1,
) -> Dict[str, Dict[str, object]]:
    """Per-neuron ERL breakdown (the advance-gate deliverable).

    Merge labels are detected **globally** (a merge is a merge even if the two
    neurons are scored separately), then each neuron's edges are scored on their
    own total length. Returns ``{neuron: erl-dict}`` (same keys as
    :func:`expected_run_length`), sorted by neuron.
    """
    edges = list(edges)
    bg = frozenset(background)
    merges = merge_labels(edges, node_label, node_neuron, background=bg,
                          min_support_frac=min_support_frac,
                          min_support_count=min_support_count)
    merge_set = set(merges)

    # bucket edges by neuron (an edge belongs to a neuron iff both ends agree)
    by_neuron: Dict[str, List[Tuple[str, str, float]]] = {}
    for u, v, length in edges:
        nu, nv = node_neuron.get(u), node_neuron.get(v)
        if nu is None or nu != nv:
            continue
        by_neuron.setdefault(nu, []).append((u, v, length))

    out: Dict[str, Dict[str, object]] = {}
    for neu in sorted(by_neuron):
        sub = by_neuron[neu]
        total = float(sum(e[2] for e in sub))
        dsu = _DSU()
        n_split = n_bg = n_merge_edges = 0
        for u, v, length in sub:
            lu, lv = node_label.get(u), node_label.get(v)
            if _is_bg(lu, bg) or _is_bg(lv, bg):
                n_bg += 1
                continue
            if lu in merge_set or lv in merge_set:
                n_merge_edges += 1
                continue
            if lu != lv:
                n_split += 1
                continue
            dsu.find(u); dsu.find(v)
            dsu.union(u, v)
            dsu.add_length(u, length)
        comp = dsu.component_lengths()
        runs = [L for L in comp.values() if L > 0]
        sum_sq = float(sum(L * L for L in runs))
        out[neu] = {
            "erl": (sum_sq / total) if total > 0 else 0.0,
            "total_length": total,
            "n_runs": len(runs),
            "max_run": float(max(runs)) if runs else 0.0,
            "n_split_edges": int(n_split),
            "n_bg_edges": int(n_bg),
            "n_merge_edges": int(n_merge_edges),
        }
    return out


def summarize(report: Dict[str, object], *, unit_nm_per: float = 1000.0) -> Dict[str, float]:
    """Convenience: convert an ERL report's nm lengths to µm (default).

    ``unit_nm_per`` = nm per output unit (1000 → µm). Returns ``erl`` and
    ``total_length`` rescaled; leaves counts alone.
    """
    return {
        "erl_um": float(report["erl"]) / unit_nm_per,
        "total_length_um": float(report["total_length"]) / unit_nm_per,
    }


# =============================================================================
# Per-node labels from a prediction  (the one drive-dependent wire-in)
# =============================================================================

def sample_node_labels(
    skel: Skeletons,
    label_slice_fn: Callable[[int], np.ndarray],
    transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    *,
    z_round: bool = True,
    radius: int = 0,
) -> Dict[str, Hashable]:
    """Assign each skeleton node the predicted label at its location.

    This is the ONLY part of ERL that needs pixel data. It is written generically
    so it tests on synthetic arrays and works later with either the GT labelmaps
    or pipeline predictions resampled onto the GT grid:

    * ``label_slice_fn(z)`` returns the 2-D label array for slice ``z`` (e.g.
      ``GroundTruth.label_slice`` for a self-consistency check, or a prediction
      store once one exists).
    * ``transform(xy, z)`` maps node **stack-px** xy at slice ``z`` to the label
      array's pixel grid, pass an :class:`eval.registration.Registration`'s
      ``.transform``; default is identity (node coords already in label-px).
    * ``radius`` (default 0 = single pixel): sample a ``(2r+1)²`` window and take the
      **dominant non-background label** (0 only if the whole window is background).
      Robust to thin structures where the exact node pixel lands just off a 1-3 px-wide
      mask (the neighborhood-sampling lever), important when the
      label grid is downscaled (``_sam``), where 1 px ≈ ``scale`` full-res px.

    A node whose mapped pixel is out of bounds gets label 0 (background).
    """
    # group nodes by integer slice so each label array is fetched once
    by_z: Dict[int, List[str]] = {}
    for nid, (x, y, z) in skel.xyz.items():
        zi = int(round(z)) if z_round else int(z)
        by_z.setdefault(zi, []).append(nid)

    labels: Dict[str, Hashable] = {}
    for zi, nodes in by_z.items():
        try:
            arr = label_slice_fn(zi)
        except (KeyError, OSError):
            for n in nodes:
                labels[n] = 0
            continue
        H, W = arr.shape[:2]
        pts = np.array([[skel.xyz[n][0], skel.xyz[n][1]] for n in nodes], dtype=float)
        if transform is not None:
            pts = transform(pts, zi)
        xc = np.round(pts[:, 0]).astype(int)
        yc = np.round(pts[:, 1]).astype(int)
        for n, xx, yy in zip(nodes, xc, yc):
            if not (0 <= xx < W and 0 <= yy < H):
                labels[n] = 0
            elif radius <= 0:
                labels[n] = int(arr[yy, xx])
            else:
                win = arr[max(0, yy - radius):yy + radius + 1,
                          max(0, xx - radius):xx + radius + 1].ravel()
                nz = win[win != 0]
                labels[n] = int(np.bincount(nz).argmax()) if nz.size else 0
    return labels
