"""gt_dataset.py, wire the real pipeline (run_chain / batch.py) onto SEM-Dauer 1.

Score the *production* pipeline against the cross-worm
GT, instead of the `predict_gt.py` reimplementation. The pipeline is worm-agnostic except
two seams, both filled here:

  1. **EM source**, `GtFrameStore` plugs into `pipeline.FrameStore`: SEM-Dauer 1's EM is a
     per-slice PNG export (`config.GT_EM_DIR`, key == VAST slice z, z 1:1), not the target
     worm's tif-by-file_z stack.
  2. **Skeleton -> image transform**, the pipeline consumes `annotate_df`'s `x_tif/y_tif`.
     For the target worm those come from the z-independent `catmaid_to_tif` affine; for p280
     they come from the **per-section** `eval.registration.Registration` (z-dependent), baked
     in here per node so every downstream phase (build_prompts, crop windows, run_qc) is
     unchanged.

`build_gt_session_inputs()` returns everything `batch._build_session` needs:
`(annotate_df, chains, frame_store)`. Output/frames roots come from the caller (batch argparse).

Registration is full-res (`registration.json` scaled x4 -> A ~ I), so `x_tif/y_tif`
land in the full_scale VAST grid (9728x9216), the same grid the full_scale EM PNGs live on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import config
from sam2_utils.skeletons import normalize_name
from .registration import Registration

_SLICE_RE = re.compile(r"_s(\d+)\.png$", re.IGNORECASE)


# =============================================================================
# EM frame source: per-slice PNG export
# =============================================================================

class GtFrameStore(pipeline.FrameStore):
    """SEM-Dauer 1 EM: per-slice PNGs ``*_s{slice:03d}.png`` under ``em_dir``.

    Logical z == VAST slice z == cache key (1:1, no FILE_Z_OFFSET). Reads go through
    the same ``cv2.imread`` / ``_read_tif_window`` the tif store uses (the windowed
    read falls back to a full decode + slice on PNG), so no read code changes.
    """

    def __init__(self, em_dir: Optional[Path] = None):
        self.em_dir = Path(em_dir) if em_dir is not None else config.GT_EM_DIR
        self._index: dict[int, Path] = {}
        for p in self.em_dir.glob("*.png"):
            m = _SLICE_RE.search(p.name)
            if m:
                self._index[int(m.group(1))] = p
        if not self._index:
            raise FileNotFoundError(f"no *_s###.png EM slices under {self.em_dir}")

    def key_of_z(self, z: int) -> int:
        return int(z)

    def z_of_key(self, key: int) -> int:
        return int(key)

    def file_for_z(self, z: int) -> Path:
        z = int(z)
        if z not in self._index:
            raise AssertionError(f"no GT EM slice for z={z} under {self.em_dir}")
        return self._index[z]

    def files_in_z_range(self, z0: int, z1: int) -> List[Tuple[int, Path]]:
        lo, hi = (int(z0), int(z1)) if z0 <= z1 else (int(z1), int(z0))
        return sorted((k, p) for k, p in self._index.items() if lo <= k <= hi)


# =============================================================================
# Skeleton table with registration-baked x_tif / y_tif
# =============================================================================

def build_gt_annotate_df(skeleton_csv: Path, registration_json: Path) -> pd.DataFrame:
    """Load the p280 node table and add `x_tif`/`y_tif` via the per-section registration.

    Each node is mapped (x, y, z)_stackpx -> (x_tif, y_tif) in the GT full-res grid using
    the affine of *its own* z-section. Vectorized per z-group (the registration is a single
    2x3 per slice), so the ~250k-node table maps in well under a second.
    """
    df = pd.read_csv(skeleton_csv)
    reg = Registration.from_json(registration_json)

    xy = df[["x", "y"]].to_numpy(dtype=float)
    z = df["z"].round().astype(int).to_numpy()
    out = np.empty_like(xy)
    for zz in np.unique(z):
        m = z == zz
        out[m] = reg.transform(xy[m], int(zz))     # reg clamps z to its fitted range
    df["x_tif"] = out[:, 0]
    df["y_tif"] = out[:, 1]
    return df


def load_gt_chains(chains_json: Path,
                   neurons: Optional[List[str]] = None,
                   neuron_limit: Optional[int] = None) -> list:
    """Load p280 chains.json, scoped to a configurable subset of neurons.

    `neurons`: explicit normalized-name allow-list (wins if given).
    `neuron_limit`: else keep the first N neurons by sorted normalized name (empty /
        unnamed fragments excluded), a quick deterministic subset without naming.
    Neither: every neuron (9766 chains, the caller should gate this; see batch.py).
    """
    with open(chains_json) as f:
        chains = json.load(f)
    if neurons is not None:
        want = {normalize_name(n) for n in neurons}
        chains = [c for c in chains if normalize_name(c["cell_name"]) in want]
    elif neuron_limit is not None:
        names = sorted({normalize_name(c["cell_name"]) for c in chains
                        if normalize_name(c["cell_name"])})
        keep = set(names[:neuron_limit])
        chains = [c for c in chains if normalize_name(c["cell_name"]) in keep]
    return chains


# =============================================================================
# One-call wiring for batch.py
# =============================================================================

def gt_paths() -> dict:
    """Canonical SEM-Dauer 1 input paths (skeletons + registration), repo-relative."""
    base = config.DATA_DIR / "groundtruth" / "skeletons_p280"
    return {
        "skeleton_csv": base / "aggregate_data_pv.csv",
        "chains_json": base / "chains.json",
        "registration_json": base / "registration.json",
        "em_dir": config.GT_EM_DIR,
    }


def build_gt_session_inputs(neurons: Optional[List[str]] = None,
                            neuron_limit: Optional[int] = None
                            ) -> Tuple[pd.DataFrame, list, GtFrameStore]:
    """Return (annotate_df, chains, frame_store) for a SEM-Dauer 1 batch run.

    `neurons` / `neuron_limit` scope the chain subset (see load_gt_chains)."""
    p = gt_paths()
    annotate_df = build_gt_annotate_df(p["skeleton_csv"], p["registration_json"])
    chains = load_gt_chains(p["chains_json"], neurons, neuron_limit)
    frame_store = GtFrameStore(p["em_dir"])
    return annotate_df, chains, frame_store
