"""score_labelmap.py, labelmap metrics (VOI, ARAND, ERL) for a SEM-Dauer 1 batch run.

Region IoU (eval.score_batch) treats each neuron independently. The connectomics
metrics need a *labelmap*: all neurons composited into one per-slice integer map, so
split/merge between objects is visible. This module builds that **in memory at the
`_sam` grid** (a full-res 9728×9216 uint16 labelmap is ~190 MP × 2 B ≈ 378 MB/slice, infeasible on disk for hundreds of slices) and computes:

  * **VOI_split / VOI_merge** and **ARAND** (pred vs GT labelmap, over GT-foreground),
  * **per-neuron ERL** + split/merge breakdown, via skeleton-node sampling through the
    registration (node coords scaled by 1/save_downscale to index the `_sam` map).

Composite convention mirrors `predict_gt.composite_labelmaps`: each neuron gets a
distinct id (1..N), **first-writer-wins** on overlaps (overlaps are the merge signal).
Tier-2 chains store masks in `_pcrop`; they are placed onto the full `_sam` frame via
the persisted `crop_window` (so the scorer is correct for both `_sam` and tier-2 runs).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from sam2_utils import config
from sam2_utils.alignment import CropWindow
from sam2_utils.skeletons import normalize_name
from . import metrics as M
from .erl import (Skeletons, expected_run_length, load_skeletons, per_neuron_erl,
                  sample_node_labels)
from .groundtruth import GroundTruth
from .registration import Registration

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# Per-chain mask -> full _sam frame (crop-aware)
# =============================================================================

def _mask_z_index(masks_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in masks_dir.glob("mask_*.png"):
        digits = "".join(c for c in p.stem if c.isdigit())
        if digits:
            out[int(digits)] = p
    return out


def chain_sam_mask(mask_path: Path, crop_window: Optional[dict],
                   sam_hw: Tuple[int, int]) -> np.ndarray:
    """Load one chain's mask at slice z and return it as a full `_sam` boolean frame.

    A non-crop (`_sam`) chain's PNG already IS the full frame. A tier-2 (`_pcrop`)
    chain's PNG is the crop; place it into the full frame via its `crop_window`.
    """
    import cv2
    m = np.asarray(Image.open(mask_path)) > 0
    if m.ndim == 3:
        m = m[0]
    H, W = sam_hw
    if crop_window is None:
        if m.shape != sam_hw:                      # defensive: resample a stray-size _sam mask
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return m
    cw = CropWindow.from_dict(crop_window)
    x0s, y0s = cw.tif_to_sam(np.asarray(cw.origin_tif, dtype=float))
    x0s, y0s = int(round(x0s)), int(round(y0s))
    w, h = cw.size_tif
    ws, hs = int(round(w / cw.sam_scale)), int(round(h / cw.sam_scale))
    full = np.zeros(sam_hw, dtype=bool)
    if ws <= 0 or hs <= 0:
        return full
    resized = cv2.resize(m.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
    # clip to frame bounds (windows are floored inside the image, but guard anyway)
    y1, x1 = min(H, y0s + hs), min(W, x0s + ws)
    y0c, x0c = max(0, y0s), max(0, x0s)
    if y1 > y0c and x1 > x0c:
        full[y0c:y1, x0c:x1] = resized[(y0c - y0s):(y1 - y0s), (x0c - x0s):(x1 - x0s)]
    return full


# =============================================================================
# Composer
# =============================================================================

class SamLabelComposer:
    """Composite a batch run's per-chain masks into per-slice `_sam` labelmaps."""

    def __init__(self, root: Path, neurons: Sequence[str], sam_hw: Tuple[int, int]):
        self.root = Path(root)
        self.sam_hw = sam_hw
        self.neurons = list(neurons)
        self.neuron_ids = {n: i + 1 for i, n in enumerate(self.neurons)}   # 1..N, 0=bg
        # neuron -> z -> [(mask_path, crop_window)]
        self._idx: Dict[str, Dict[int, List[Tuple[Path, Optional[dict]]]]] = \
            defaultdict(lambda: defaultdict(list))
        for neu in self.neurons:
            nd = self.root / neu
            if not nd.is_dir():
                continue
            for ch in sorted(nd.glob("chain_*")):
                md = ch / "masks"
                if not md.is_dir():
                    continue
                cw = None
                st = ch / "state.json"
                if st.exists():
                    try:
                        cw = json.loads(st.read_text()).get("crop_window")
                    except Exception:
                        cw = None
                for z, p in _mask_z_index(md).items():
                    self._idx[neu][z].append((p, cw))

    def slices(self) -> List[int]:
        zs = set()
        for neu in self._idx:
            zs.update(self._idx[neu])
        return sorted(zs)

    def neuron_mask(self, neuron: str, z: int) -> Optional[np.ndarray]:
        items = self._idx.get(neuron, {}).get(int(z), [])
        if not items:
            return None
        union = np.zeros(self.sam_hw, dtype=bool)
        for p, cw in items:
            union |= chain_sam_mask(p, cw, self.sam_hw)
        return union

    def labelmap(self, z: int) -> Tuple[np.ndarray, int]:
        """Per-slice `_sam` uint16 labelmap (neuron->id, first-writer-wins). +collisions."""
        lab = np.zeros(self.sam_hw, dtype=np.uint16)
        written = np.zeros(self.sam_hw, dtype=bool)
        collisions = 0
        for neu in self.neurons:                       # id order == first-writer order
            m = self.neuron_mask(neu, z)
            if m is None:
                continue
            collisions += int((m & written).sum())
            take = m & ~written
            lab[take] = self.neuron_ids[neu]
            written |= m
        return lab, collisions


# =============================================================================
# GT at the _sam grid
# =============================================================================

def gt_sam_labelmap(gt: GroundTruth, z: int, sam_hw: Tuple[int, int],
                    scored_nrs: Optional[set] = None) -> np.ndarray:
    """GT labelmap downscaled (nearest) to the `_sam` grid; optionally restricted to
    `scored_nrs` (others -> 0) so VOI/ARAND score only the measured neurons."""
    import cv2
    full = gt.label_slice(z)
    if scored_nrs is not None:
        full = np.where(np.isin(full, list(scored_nrs)), full, 0)
    H, W = sam_hw
    return cv2.resize(full.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# The labelmap score
# =============================================================================

def score_labelmap(
    root: Path,
    gt: GroundTruth,
    neurons: Sequence[str],
    *,
    skeleton_csv: Path,
    registration_json: Path,
    save_downscale: int = 8,
    merge_tol_frac: float = 0.1,
    merge_tol_count: int = 1,
    node_sample_radius: int = 2,
    progress: bool = True,
) -> dict:
    """Composite + VOI/ARAND + per-neuron ERL for a SEM-Dauer 1 batch run."""
    Hf, Wf = gt.label_slice(next(iter(gt.slice_indices))).shape
    sam_hw = (Hf // save_downscale, Wf // save_downscale)
    comp = SamLabelComposer(root, neurons, sam_hw)
    scored = [n for n in neurons if gt.nr_for_label(n)]
    scored_nrs = set()
    for n in scored:
        scored_nrs.update(gt.nr_for_label(n))

    # --- VOI + ARAND over GT-foreground, accumulated across slices (volume-level) ---
    t0 = time.perf_counter()
    pred_fg: List[np.ndarray] = []
    gt_fg: List[np.ndarray] = []
    collisions = 0
    sl = comp.slices()
    if progress:
        print(f"[labelmap] compositing + VOI/ARAND over {len(sl)} slices "
              f"(_sam grid {sam_hw})", flush=True)
    for i, z in enumerate(sl, 1):
        if not gt.has_slice(z):
            continue
        lab, c = comp.labelmap(z)
        collisions += c
        gtl = gt_sam_labelmap(gt, z, sam_hw, scored_nrs)
        fg = gtl > 0
        if fg.any():
            pred_fg.append(lab[fg])
            gt_fg.append(gtl[fg])
        if progress and i % 50 == 0:
            el = time.perf_counter() - t0
            print(f"      {i}/{len(sl)} slices  {i/el:.2f} slice/s  elapsed {el:.0f}s",
                  flush=True)
    voi_arand = None
    if pred_fg:
        p = np.concatenate(pred_fg); g = np.concatenate(gt_fg)
        # skimage refs by default (CAD/FGNet methodology); inputs already GT-foreground
        # (== ignore_labels=(0,)). Falls back to the numpy metrics if skimage is absent.
        voi_arand = M.voi_arand(p, g)
    voi_secs = time.perf_counter() - t0

    # --- per-neuron ERL (node sampling through registration, scaled to _sam) ---
    t1 = time.perf_counter()
    skel = load_skeletons(skeleton_csv)
    keep = set(scored)
    skel = Skeletons(
        edges=[(u, v, L) for (u, v, L) in skel.edges
               if normalize_name(skel.neuron.get(u, "")) in keep
               and normalize_name(skel.neuron.get(v, "")) in keep],
        xyz={n: p for n, p in skel.xyz.items()
             if normalize_name(skel.neuron.get(n, "")) in keep},
        neuron={n: normalize_name(skel.neuron[n]) for n in skel.xyz
                if normalize_name(skel.neuron.get(n, "")) in keep},
    )
    reg = Registration.from_json(registration_json)
    ds = float(save_downscale)
    label_fn: Callable[[int], np.ndarray] = lambda z: comp.labelmap(z)[0]
    transform = lambda xy, z: reg.transform(np.asarray(xy, dtype=float), z) / ds
    if progress:
        print(f"[labelmap] ERL: sampling {len(skel.xyz)} nodes over "
              f"{len(set(skel.neuron.values()))} neurons", flush=True)
    sampled = sample_node_labels(skel, label_fn, transform=transform,
                                 radius=node_sample_radius)
    node_label: Dict[str, Hashable] = {n: (int(v) if v else 0) for n, v in sampled.items()}
    erl = expected_run_length(skel.edges, node_label, skel.neuron,
                              min_support_frac=merge_tol_frac, min_support_count=merge_tol_count)
    per_neuron = per_neuron_erl(skel.edges, node_label, skel.neuron,
                                min_support_frac=merge_tol_frac, min_support_count=merge_tol_count)
    ceil = expected_run_length(skel.edges, dict(skel.neuron), skel.neuron)
    erl_secs = time.perf_counter() - t1

    return {
        "sam_hw": list(sam_hw), "n_slices": len(sl), "overlap_collisions_px": collisions,
        "neuron_ids": comp.neuron_ids,
        "voi": (None if voi_arand is None else
                {k: voi_arand[k] for k in ("voi_split", "voi_merge", "voi")}),
        "arand": (None if voi_arand is None else
                  {"are": voi_arand["are"], "precision": voi_arand["arand_precision"],
                   "recall": voi_arand["arand_recall"]}),
        "metric_backend": (None if voi_arand is None else voi_arand["backend"]),
        "erl_um": erl["erl"] / 1000.0, "ceiling_um": ceil["erl"] / 1000.0,
        "pct_of_ceiling": (erl["erl"] / ceil["erl"] * 100.0) if ceil["erl"] else 0.0,
        "n_merge_labels": erl.get("n_merge_labels"), "n_split_edges": erl.get("n_split_edges"),
        "n_bg_edges": erl.get("n_bg_edges"), "n_nodes": len(sampled),
        "merge_tol_frac": merge_tol_frac, "node_sample_radius": node_sample_radius,
        "per_neuron_erl_um": {n: round(d["erl"] / 1000.0, 3) for n, d in per_neuron.items()},
        "timing": {"voi_arand_seconds": round(voi_secs, 1), "erl_seconds": round(erl_secs, 1)},
    }
