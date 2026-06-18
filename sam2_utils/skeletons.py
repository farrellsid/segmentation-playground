"""
skeletons.py — pull + process CATMAID skeletons into the pipeline's node table.

This is the library form of `Copy_of_Get_catmaid_information_lucinda.ipynb`: the
notebook that produced `data/aggregate_data_pv.csv` + `chains.json` + `roots.json`
for the original worm (CATMAID project 336). Factored into reusable functions so a
new worm (e.g. the cross-worm GT, project 280) is one driver call — see
`pull_worm.py`.

The pipeline (three stages, each a function here)
-------------------------------------------------
1. **pull_aggregate** — every skeleton's node-overview into one DataFrame, with
   nm→stack-px conversion (÷ STACK_RESOLUTION_NM). `node_id` is kept exactly as
   CATMAID returns it (integer); downstream matching everywhere stringifies via
   ``annotate_df["node_id"].astype(str) == str(node)`` (pipeline.py §"node_id
   matching"), so CSV and chains.json need only be *internally* consistent.
2. **insert_virtual_nodes** — CATMAID traces skip z-sections; SAM2 propagation
   needs a node on every slice. For each parent→child edge spanning a z-gap (up to
   ``allowance`` sections) insert linearly-interpolated "virtual" nodes, one per
   intervening slice, and rewire the chain child → v1 → … → vk → parent. Returns
   the densified table with an ``is_vnode`` column.
3. **decompose_chains** — split each neuron's (possibly branched) skeleton tree at
   every branch point into maximal *linear* chains (a DFS). Each chain is what the
   pipeline propagates as one unit. Returns (chains, roots).

Note on the vnode wiring vs the notebook
-----------------------------------------
The notebook wired vnode parents in a per-chain Python loop (~19 min on the
original worm) whose ``.loc[isin]`` assignment was order-fragile for downward
z-gaps. Here it's a vectorized groupby-shift: child → nearest-to-child vnode,
each vnode → the next one toward the parent, last → the original parent. Same
intent, correct for both gap directions, and seconds not minutes.

Torch-free; only requests (via :mod:`sam2_utils.catmaid`) + pandas/numpy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .catmaid import Catmaid

# node-overview column order (CATMAID returns rows in this order).
NODE_COLS = ["node_id", "parent_id", "confidence", "x", "y", "z",
             "radius", "creator_id", "edition_time"]


def normalize_name(raw: str) -> str:
    """CATMAID cell_name -> bare identity, to join skeletons to the VAST GT labels.

    CATMAID names carry decorations the VAST metadata strips: surrounding
    brackets (``[IL2L]``) and trailing confirmation marks (``RMDVR!``, ``URYDR?``).
    Mirror :func:`eval.groundtruth.parse_name`'s label so the two tables join.
    """
    s = str(raw).strip()
    if s.startswith("[") and "]" in s:
        s = s[1:s.rindex("]")]
    return s.strip().rstrip("!?").strip()


# =============================================================================
# 1. Pull
# =============================================================================

def pull_aggregate(
    catmaid: Catmaid,
    *,
    resolution_nm: Sequence[float] = config.STACK_RESOLUTION_NM,
    skeleton_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Pull every skeleton's nodes into one DataFrame, nm→stack-px on x/y/z.

    Mirrors the notebook's aggregate-build cell exactly (keeps ``node_id`` as the
    integer CATMAID returns; only x/y/z are scaled). Pass ``skeleton_names`` to
    reuse an already-fetched id→name map.
    """
    if skeleton_names is None:
        skels = catmaid.get_skeletons()
        skeleton_names = catmaid.load_skeleton_names(skels)
    rx, ry, rz = resolution_nm

    frames: List[pd.DataFrame] = []
    for code in tqdm(skeleton_names, desc="node overviews"):
        rows = catmaid.node_overview(code)[0]
        if not rows:
            continue
        d = pd.DataFrame(rows, columns=NODE_COLS)
        d["cell_name"] = skeleton_names[code]
        d["x"] = d["x"] / rx
        d["y"] = d["y"] / ry
        d["z"] = d["z"] / rz
        frames.append(d)

    agg = pd.concat(frames, ignore_index=True)
    print(f"[skeletons] {len(skeleton_names)} skeletons, {len(agg)} nodes")
    return agg


# =============================================================================
# 2. Virtual nodes
# =============================================================================

def insert_virtual_nodes(agg: pd.DataFrame, *, allowance: int = 50) -> pd.DataFrame:
    """Densify z-gaps with interpolated virtual nodes; return the bigger table.

    For each parent→child edge whose ``|z_parent - z_child| - 1`` gap is in
    ``[1, allowance]``, insert one vnode per intervening z-section, linearly
    interpolating x/y, and rewire ``child → v(nearest child) → … → parent``.
    vnode ``node_id`` is ``"v_<child_node_id>_<layer>"``; vnodes carry
    ``is_vnode=True`` and NaN for the non-geometry columns. Original rows get
    ``is_vnode=False`` (and the gap-children get their ``parent_id`` repointed at
    their first vnode).
    """
    df = agg.copy()

    # node_id -> (z, x, y) lookups, keyed numerically so int node_id matches a
    # float parent_id (CATMAID hands roots a NaN parent, which floats the column).
    key = pd.to_numeric(df["node_id"], errors="coerce")
    z = pd.to_numeric(df["z"], errors="coerce")
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    lut_z = dict(zip(key, z)); lut_x = dict(zip(key, x)); lut_y = dict(zip(key, y))

    pk = pd.to_numeric(df["parent_id"], errors="coerce")
    e = pd.DataFrame({
        "node_id": df["node_id"].values, "cell_name": df["cell_name"].values,
        "parent_id": df["parent_id"].values,
        "z": z.values, "x": x.values, "y": y.values,
        "z_parent": pk.map(lut_z).values,
        "x_parent": pk.map(lut_x).values,
        "y_parent": pk.map(lut_y).values,
    }).dropna(subset=["z_parent"])

    gap = e["z_parent"] - e["z"] - 1
    e["gap_abs"] = gap.abs()
    e["gap_sign"] = np.where(gap >= 0, 1, -1)
    cand = e[(e["gap_abs"] > 0) & (e["gap_abs"] <= allowance)].copy()
    if cand.empty:
        df["is_vnode"] = False
        return df

    # one row per intervening z-layer, ordered from child toward parent
    cand["layers"] = cand.apply(
        lambda r: list(range(int(r["z"]) + int(r["gap_sign"]),
                             int(r["z_parent"]), int(r["gap_sign"]))), axis=1)
    ex = cand.explode("layers").dropna(subset=["layers"]).copy()
    ex["layers"] = ex["layers"].astype(int)
    ex["rank"] = ex.groupby("node_id").cumcount() + 1     # 1 == nearest the child
    ex["vnode_id"] = "v_" + ex["node_id"].astype(str) + "_" + ex["layers"].astype(str)

    span = ex["gap_abs"] + 1
    ex["x_v"] = ex["x"] + ex["rank"] * (ex["x_parent"] - ex["x"]) / span
    ex["y_v"] = ex["y"] + ex["rank"] * (ex["y_parent"] - ex["y"]) / span

    # parent of each vnode = the next vnode toward the parent; the last vnode's
    # parent = the edge's original parent_id.
    ex = ex.sort_values(["node_id", "rank"])
    nxt = ex.groupby("node_id")["vnode_id"].shift(-1)
    ex["vnode_parent"] = nxt.where(nxt.notna(), ex["parent_id"])

    vnodes = pd.DataFrame({
        "node_id": ex["vnode_id"].values,
        "parent_id": ex["vnode_parent"].values,
        "x": ex["x_v"].values, "y": ex["y_v"].values, "z": ex["layers"].values,
        "cell_name": ex["cell_name"].values, "is_vnode": True,
    })
    for col in df.columns:
        if col not in vnodes.columns:
            vnodes[col] = np.nan

    # repoint each gap-child at its nearest (rank-1) vnode
    first = ex.loc[ex["rank"] == 1].set_index("node_id")["vnode_id"]
    df["is_vnode"] = False
    df["parent_id"] = df["node_id"].map(first).where(
        df["node_id"].isin(first.index), df["parent_id"])

    out = pd.concat([df, vnodes[df.columns]], ignore_index=True)
    out = out.sort_values("z").reset_index(drop=True)
    print(f"[skeletons] inserted {len(vnodes)} virtual nodes "
          f"(allowance={allowance}); total {len(out)} nodes")
    return out


# =============================================================================
# 3. Decompose into linear chains
# =============================================================================

def decompose_chains(
    agg_pv: pd.DataFrame,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, str]]:
    """Split branched skeletons into maximal linear chains (DFS at branch points).

    ``start`` / ``end`` clip to the imaged z-range first (inclusive); None = no
    clip on that side. Returns ``(chains, roots)``:
      chains : list of {cell_name, nodes:[node_id,...], origin_branch, is_root}
      roots  : {node_id: cell_name} for every node whose parent is out of scope.

    Node ids are coerced to JSON-native (python int / str) so the result dumps
    cleanly and round-trips to the same string the pipeline matches on.
    """
    df = agg_pv
    if start is not None:
        df = df[df["z"] >= start]
    if end is not None:
        df = df[df["z"] <= end]

    def _nat(v):
        """numpy/pandas scalar -> JSON-native; integer-valued floats -> int."""
        if isinstance(v, str):
            return v
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return int(v) if float(v).is_integer() else float(v)
        return v

    node_ids = [_nat(n) for n in df["node_id"].tolist()]
    parents = [_nat(p) for p in df["parent_id"].tolist()]
    names = df["cell_name"].tolist()
    all_nodes = set(node_ids)
    name_of = dict(zip(node_ids, names))
    parent_of = dict(zip(node_ids, parents))

    children: Dict[object, List[object]] = defaultdict(list)
    for nid, pid in zip(node_ids, parents):
        if pid in all_nodes:
            children[pid].append(nid)

    # a node whose parent is out of scope (None, or clipped away) starts a tree
    roots = {n: name_of[n] for n in node_ids if parent_of[n] not in all_nodes}

    chains: List[dict] = []
    stack = [(root, [root], None, cell_name) for root, cell_name in roots.items()]
    with tqdm(total=len(node_ids), desc="decompose") as pbar:
        while stack:
            node, current, origin, cell_name = stack.pop()
            kids = children.get(node, [])
            if len(kids) == 0:
                chains.append({"cell_name": cell_name, "nodes": current,
                               "origin_branch": origin, "is_root": origin is None})
            elif len(kids) == 1:
                stack.append((kids[0], current + [kids[0]], origin, cell_name))
            else:
                chains.append({"cell_name": cell_name, "nodes": current,
                               "origin_branch": origin, "is_root": origin is None})
                for kid in kids:
                    stack.append((kid, [kid], node, cell_name))
            pbar.update(1)
            pbar.set_postfix(chains=len(chains))

    # roots keys must be JSON strings (json turns int keys to str anyway)
    roots_out = {str(k): v for k, v in roots.items()}
    print(f"[skeletons] {len(chains)} chains from {len(roots)} roots "
          f"(z in [{start}, {end}])")
    return chains, roots_out
