"""registration_overlay.py, the Stage 0.1 human gut-check for the SEM-Dauer 1 registration.

The per-section affine that maps the CATMAID project-280 skeleton onto the VAST grid was
validated quantitatively (on-mask 67.9% -> 85.7% at quarter scale, 91.7% at full res). This
module is the remaining piece: an interactive napari overlay so a human can scrub the full-res
VAST EM and confirm the registered nodes actually sit on the right neurites.

It overlays two point layers on the EM:

  * **raw nodes**, each skeleton node at its untransformed CATMAID stack-px (x, y);
  * **registered nodes**, the same nodes pushed through the registration to VAST full-res px
    (x_tif, y_tif).

The registration's linear part is ~ identity at full res (same scale and orientation), but it also
applies a per-section translation of order ~100-250 px, so the raw nodes sit that far off the
neurite while the registered nodes should land on it; the gap between the two layers is the
realignment the registration is correcting for. Clicking the EM prints the clicked coordinate and
the nearest node's name plus its raw CATMAID
(x, y, z), the numbers to type into the CATMAID web client to find the same spot. CATMAID itself
is not queried; the user does that side by hand.

Run:
    py -3 -m eval.registration_overlay --start-z 400

The data helpers (`build_overlay_table`, `nodes_on_slice`, `nearest_node`) are pure and
torch/napari-free so they unit-test without a display; napari and dask are imported lazily in
`launch`.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# `eval.gt_dataset` pulls in `pipeline` (torch), so it is imported lazily inside the functions
# that need it; this keeps the module top-level and the pure helpers below torch/napari-free.

# Columns surfaced for the click readout (the CATMAID-side coordinates).
_META_COLS = ["node_id", "cell_name", "x", "y", "z"]


# =============================================================================
# Pure data helpers (no napari, no torch)
# =============================================================================

def _add_overlay_columns(df: pd.DataFrame, include_vnodes: bool = True) -> pd.DataFrame:
    """Add the integer `z_int` column and optionally drop virtual nodes. Pure."""
    df = df.copy()
    df["z_int"] = df["z"].round().astype(int)
    if not include_vnodes and "is_vnode" in df.columns:
        df = df[~df["is_vnode"].astype(bool)].reset_index(drop=True)
    return df


def build_overlay_table(skeleton_csv: Path, registration_json: Path,
                        include_vnodes: bool = True) -> pd.DataFrame:
    """The p280 node table with raw (x, y, z) and registered (x_tif, y_tif) coords.

    Thin wrapper over `eval.gt_dataset.build_gt_annotate_df` that adds an integer `z_int`
    column (the VAST slice each node lands on) for fast per-slice filtering, and optionally
    drops the interpolated virtual nodes (`is_vnode`), which carry no independent registration
    information.
    """
    from .gt_dataset import build_gt_annotate_df
    df = build_gt_annotate_df(Path(skeleton_csv), Path(registration_json))
    return _add_overlay_columns(df, include_vnodes=include_vnodes)


def nodes_on_slice(df: pd.DataFrame, z: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Raw and registered point arrays for the nodes on VAST slice `z`.

    Returns `(raw_pts, reg_pts, meta)` where both arrays are `(N, 3)` shaped for napari as
    `(plane, row, col) == (z, y, x)`: `raw_pts` uses the untransformed `(y, x)`, `reg_pts`
    uses the registered `(y_tif, x_tif)`. The plane coordinate is `z` for both so they render
    on the current slice. `meta` is the per-node readout frame aligned row-for-row with the
    point arrays.
    """
    z = int(z)
    sub = df[df["z_int"] == z]
    n = len(sub)
    plane = np.full(n, z, dtype=float)
    raw_pts = np.column_stack([plane, sub["y"].to_numpy(float), sub["x"].to_numpy(float)])
    reg_pts = np.column_stack([plane, sub["y_tif"].to_numpy(float), sub["x_tif"].to_numpy(float)])
    meta = sub[_META_COLS].reset_index(drop=True)
    return raw_pts, reg_pts, meta


def nearest_node(meta: pd.DataFrame, reg_pts: np.ndarray, click_yx: Tuple[float, float],
                 radius_px: float) -> Optional[dict]:
    """The registered node nearest to a click, within `radius_px`, or None.

    `click_yx` and the registered points are both in full-res VAST px. Returns the node's
    readout fields (name, id, raw CATMAID x/y/z) plus the click distance.
    """
    if len(reg_pts) == 0:
        return None
    dy = reg_pts[:, 1] - float(click_yx[0])
    dx = reg_pts[:, 2] - float(click_yx[1])
    dist = np.hypot(dy, dx)
    i = int(np.argmin(dist))
    if dist[i] > float(radius_px):
        return None
    row = meta.iloc[i]
    return {
        "node_id": row["node_id"],
        "cell_name": row["cell_name"],
        "catmaid_x": float(row["x"]),
        "catmaid_y": float(row["y"]),
        "catmaid_z": int(round(float(row["z"]))),
        "dist_px": float(dist[i]),
    }


# =============================================================================
# napari overlay
# =============================================================================

def launch(start_z: int = 0, point_size: float = 15.0, include_vnodes: bool = True,
           show_gt_mask: bool = False, block: bool = True):
    """Open the napari overlay. See module docstring.

    `point_size` is in full-res VAST px; `show_gt_mask` adds the VAST segment labelmap as a
    Labels layer (opt-in, heavier). Returns the napari viewer.
    """
    import napari
    import dask.array as da
    from dask import delayed
    from PIL import Image

    from .gt_dataset import GtFrameStore, gt_paths

    Image.MAX_IMAGE_PIXELS = None     # full-res slice is ~89.6M px, just over PIL's default

    paths = gt_paths()
    print(f"[overlay] loading skeleton table from {paths['skeleton_csv'].name} ...")
    df = build_overlay_table(paths["skeleton_csv"], paths["registration_json"],
                             include_vnodes=include_vnodes)
    print(f"[overlay] {len(df)} nodes over z [{df['z_int'].min()}, {df['z_int'].max()}]")

    # Lazy full-res EM stack indexed by absolute VAST z (plane index == z == CATMAID z).
    fs = GtFrameStore(paths["em_dir"])
    z_to_path = dict(fs.files_in_z_range(0, 10 ** 9))
    z_max = max(z_to_path)
    sample = np.asarray(Image.open(z_to_path[min(z_to_path)]))
    h, w = sample.shape[:2]
    blank = np.zeros((h, w), dtype=sample.dtype)

    @lru_cache(maxsize=8)
    def _read_em(z: int) -> np.ndarray:
        p = z_to_path.get(int(z))
        return blank if p is None else np.asarray(Image.open(p))

    lazy = [da.from_delayed(delayed(_read_em)(z), shape=(h, w), dtype=sample.dtype)
            for z in range(z_max + 1)]
    em_stack = da.stack(lazy, axis=0)

    viewer = napari.Viewer(title="SEM-Dauer 1 registration overlay (Stage 0.1)")
    viewer.add_image(em_stack, name="EM", contrast_limits=[0, 255], multiscale=False)

    if show_gt_mask:
        from .groundtruth import GroundTruth
        gt = GroundTruth.from_config()
        lazy_lbl = [da.from_delayed(delayed(gt.label_slice)(z), shape=(h, w), dtype=np.uint16)
                    for z in range(z_max + 1)]
        viewer.add_labels(da.stack(lazy_lbl, axis=0), name="GT segments", opacity=0.4)

    raw_layer = viewer.add_points(
        np.empty((0, 3)), name="raw nodes", ndim=3, size=point_size,
        face_color="cyan", border_color="black", opacity=0.6)
    reg_layer = viewer.add_points(
        np.empty((0, 3)), name="registered nodes", ndim=3, size=point_size,
        face_color="magenta", border_color="black", opacity=0.6)
    raw_layer.editable = False
    reg_layer.editable = False

    # Per-slice refresh: keep each layer at the current z's nodes (~hundreds), not all 254k.
    state: dict = {"meta": df.iloc[:0][_META_COLS], "reg_pts": np.empty((0, 3)), "z": -1}

    def _refresh(*_a) -> None:
        z = int(viewer.dims.current_step[0])
        if z == state["z"]:
            return
        raw_pts, reg_pts, meta = nodes_on_slice(df, z)
        raw_layer.data = raw_pts
        reg_layer.data = reg_pts
        state.update(meta=meta, reg_pts=reg_pts, z=z)

    viewer.dims.events.current_step.connect(_refresh)

    radius_px = max(point_size * 2.0, 30.0)

    @viewer.mouse_drag_callbacks.append
    def _on_click(_viewer, event):       # click (no drag) -> print coords + nearest node
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged or len(event.position) < 3:
            return
        z, y, x = (int(round(event.position[0])), float(event.position[1]),
                   float(event.position[2]))
        print(f"[overlay] click @ VAST full-res px: z={z} y={y:.1f} x={x:.1f}")
        hit = nearest_node(state["meta"], state["reg_pts"], (y, x), radius_px)
        if hit is None:
            print(f"           (no node within {radius_px:.0f} px on this slice)")
        else:
            print(f"           nearest node: {hit['cell_name']} (id {hit['node_id']}), "
                  f"{hit['dist_px']:.1f} px away; CATMAID stack x={hit['catmaid_x']:.1f} "
                  f"y={hit['catmaid_y']:.1f} z={hit['catmaid_z']}")

    viewer.dims.set_current_step(0, int(np.clip(start_z, 0, z_max)))
    _refresh()
    print("[overlay] cyan = raw CATMAID coords, magenta = registered; click EM for a readout.")
    if block:
        napari.run()
    return viewer


def main() -> None:
    ap = argparse.ArgumentParser(description="SEM-Dauer 1 registration overlay (Stage 0.1 gut-check)")
    ap.add_argument("--start-z", type=int, default=0, help="VAST slice to open on")
    ap.add_argument("--point-size", type=float, default=15.0, help="node marker diameter (full-res px)")
    ap.add_argument("--no-vnodes", action="store_true", help="hide interpolated virtual nodes")
    ap.add_argument("--show-gt-mask", action="store_true",
                    help="also overlay the VAST segment labelmap (heavier)")
    args = ap.parse_args()
    launch(start_z=args.start_z, point_size=args.point_size,
           include_vnodes=not args.no_vnodes, show_gt_mask=args.show_gt_mask)


if __name__ == "__main__":
    main()
