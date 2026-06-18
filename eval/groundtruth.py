"""
groundtruth.py — read the cross-worm VAST ground truth (eval/ Stage 0).

The June-2026 step-back obtained manual VAST segmentation for a *different* worm
(SEM dauer 1) with matching EM — the first real ruler for the pipeline
(PIPELINE_CONTEXT §"Research step-back"; FUTURE_DIRECTIONS §3, §4.1). This module
is the read layer for that GT: parse the VAST metadata table, and load per-slice
binary masks for any segment / neuron. Torch-free / cv2-free (numpy + Pillow +
pandas only), like ``sam2_utils.alignment`` / ``qc`` / ``labels`` — so it exercises
on any box and stays cheap to import.

What the VAST export actually is
--------------------------------
Each ``*.vsseg_export_s###.png`` is a **16-bit single-channel labelmap** (PIL mode
``I;16``), NOT an RGB color image: every pixel's value is the segment number
(``Nr``) from the metadata, with ``0`` == Background. They look near-black because
IDs are small (≤450). So the GT mask for segment ``Nr`` on slice ``s`` is simply
``label_slice(s) == Nr`` — no palette lookup, no color matching.

The metadata file (``VAST_segmentation_metadata.txt``) is VAST-Lite's "extended
segmentation color file". One row per segment::

    Nr  flags  r1 g1 b1 p1  r2 g2 b2 p2  ax ay az  parent child prev next  collapsed
        bx1 by1 bz1 bx2 by2 bz2  "name"

24 integer fields then a quoted name. We keep the structural columns (Nr, the
hierarchy links, the anchor, the bbox) and parse ``name`` into a neuron identity:

    "RMDVR_3000154.swc"        -> label "RMDVR",        bracketed=False, skeleton_id=3000154
    "[IL2L]_2998706.swc"       -> label "IL2L",         bracketed=True,  skeleton_id=2998706
    "[PVNL_or_R_1]_2998457.swc"-> label "PVNL_or_R_1",  bracketed=True,  skeleton_id=2998457
    "phagosome62"              -> label "phagosome62",  bracketed=False, skeleton_id=None
    "Background" (Nr 0)        -> label "Background",    kind="background"

Per the lab: **everything present in this file is a manually-confirmed segment**
(only confirmed objects were imported), so there is no separate "confirmed" flag
to filter on — every named segment is valid GT. ``kind`` ("neuron" / "organelle" /
"background") is a *convenience* classifier (heuristic, by name) for callers who
want to score neurons only; it is not a confirmation gate. ``bracketed`` is exposed
raw in case the bracket convention turns out to carry meaning later.

Coordinate note (full-res vs export)
-------------------------------------
The bbox/anchor coordinates in the metadata are in **full-res VAST pixels**; the
exported masks/EM are downscaled ``config.GT_DOWNSCALE`` × (default 4). Use
:meth:`GroundTruth.bbox_in_mask_px` to get a bbox in the mask's own pixel grid.

Slice index vs pipeline z
-------------------------
GT slices are numbered ``s000``..``s850`` (the VAST stack's own section index).
The pipeline's predicted masks are named by ``catmaid_z`` in the *target* worm's
frame — a different stack with no established mapping into this one. This module
deliberately does **not** invent that mapping: it indexes everything by the VAST
slice index, and the eventual prediction-vs-GT alignment is the score.py wire-in
point (and the open xy/z-registration question, see eval/README.md).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

# The full-res GT export (Stage 0.3) is 9728×9216 = 89.6M px, just over PIL's default
# decompression-bomb limit (~89.5M), so every slice read would otherwise warn. These
# are trusted local files, so lift the cap.
Image.MAX_IMAGE_PIXELS = None

try:                                   # importable without the package installed
    from sam2_utils import config
except Exception:                      # pragma: no cover - fallback for odd CWDs
    config = None


# Name fragments that mark a non-neuron object. Substring match, lower-cased.
# Heuristic only (see module docstring): used by `kind`, never to drop GT.
_ORGANELLE_TERMS = ("phagosome", "fragment", "frag", "vesicle", "dying_cell")

# `name` tail: "_<skeletonid>.swc". Captured so we can recover the SWC skeleton id
# (the ERL hook for a later stage) and strip it to get the bare identity.
_SWC_TAIL = re.compile(r"_(\d+)\.swc$")

# slice index embedded in an export filename: "..._s007.png" -> 7
_SLICE_IDX = re.compile(r"_s(\d+)\.png$", re.IGNORECASE)

# The 24 integer columns of a VAST extended-segmentation row, in file order.
_NUM_COLS = [
    "nr", "flags",
    "r1", "g1", "b1", "p1", "r2", "g2", "b2", "p2",
    "anchor_x", "anchor_y", "anchor_z",
    "parent", "child", "prev", "next", "collapsed",
    "bbox_x1", "bbox_y1", "bbox_z1", "bbox_x2", "bbox_y2", "bbox_z2",
]


def parse_name(raw: str) -> Tuple[str, bool, Optional[int]]:
    """Split a VAST segment name into ``(label, bracketed, skeleton_id)``.

    label       : bare identity (brackets + ``_<id>.swc`` tail removed).
    bracketed   : whether the original name was wrapped in ``[...]``.
    skeleton_id : the SWC skeleton id from the tail, or None.
    """
    name = raw.strip()
    bracketed = name.startswith("[")

    skeleton_id: Optional[int] = None
    m = _SWC_TAIL.search(name)
    if m:
        skeleton_id = int(m.group(1))
        name = name[: m.start()]
    elif name.endswith(".swc"):
        name = name[:-4]

    name = name.strip()
    if name.startswith("[") and "]" in name:
        name = name[1: name.rindex("]")]
    return name.strip(), bracketed, skeleton_id


def _classify(nr: int, label: str) -> str:
    """Coarse, heuristic object class from (nr, label). Convenience only."""
    if nr == 0:
        return "background"
    low = label.lower()
    if any(term in low for term in _ORGANELLE_TERMS):
        return "organelle"
    return "neuron"


def parse_metadata(path: Union[str, Path]) -> pd.DataFrame:
    """Parse a VAST extended-segmentation-color file into a DataFrame.

    One row per segment, indexed by ``nr``. Columns: the 24 structural integers
    (see ``_NUM_COLS``) plus ``name`` (raw), ``label`` / ``bracketed`` /
    ``skeleton_id`` (from :func:`parse_name`), and ``kind``.

    Comment lines (leading ``%``) and blanks are skipped.
    """
    rows: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        # name is the quoted tail; the head is the 24 whitespace-separated ints.
        q = line.find('"')
        if q == -1:
            continue
        head, name = line[:q], line[q:].strip().strip('"')
        parts = head.split()
        if len(parts) < len(_NUM_COLS):
            continue
        vals = [int(x) for x in parts[: len(_NUM_COLS)]]
        row = dict(zip(_NUM_COLS, vals))
        label, bracketed, skel = parse_name(name)
        row.update(name=name, label=label, bracketed=bracketed,
                   skeleton_id=skel, kind=_classify(row["nr"], label))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no segment rows parsed from {path}")
    return df.set_index("nr", drop=False).sort_index()


def _read_label_png(path: Path, *, retries: int = 4) -> np.ndarray:
    """Read a VAST labelmap PNG as a 2-D ``uint16`` array (pixel value == Nr).

    Retries transient read failures: the GT lives on an external drive that can
    hiccup mid-sweep (the pipeline guards against the same; batch.py "sudden unplug
    of drive"). Backs off briefly and re-opens before giving up.
    """
    import time
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with Image.open(path) as im:
                im.load()
                arr = np.asarray(im)
            if arr.ndim != 2:
                raise ValueError(
                    f"{path.name}: expected single-channel labelmap, got shape {arr.shape}")
            # PIL may hand back int32 for mode 'I'; ids are small, so a safe cast.
            return arr if arr.dtype == np.uint16 else arr.astype(np.uint16)
        except (FileNotFoundError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise OSError(f"failed to read {path} after {retries} retries: {last}")


@dataclass
class GroundTruth:
    """Read access to one cross-worm VAST GT volume.

    Build with :meth:`from_config` (uses ``sam2_utils.config`` paths) or
    :meth:`load`. Everything is indexed by the **VAST slice index** (the ``s###``
    in the filenames), exposed as :attr:`slice_indices`.

    Cheap to construct (only the metadata is parsed up front); label slices are
    read lazily and a few most-recent ones are cached.
    """

    metadata: pd.DataFrame
    mask_dir: Path
    em_dir: Optional[Path] = None
    downscale: int = 4
    _mask_files: "OrderedDict[int, Path]" = field(default_factory=OrderedDict, repr=False)
    _em_files: "OrderedDict[int, Path]" = field(default_factory=OrderedDict, repr=False)
    _cache_cap: int = 4
    _label_cache: "OrderedDict[int, np.ndarray]" = field(default_factory=OrderedDict, repr=False)
    _by_label: Dict[str, List[int]] = field(default_factory=dict, repr=False)

    # -- construction ----------------------------------------------------------
    @classmethod
    def load(
        cls,
        metadata_path: Union[str, Path],
        mask_dir: Union[str, Path],
        em_dir: Optional[Union[str, Path]] = None,
        *,
        downscale: int = 4,
        mask_glob: str = "*.vsseg_export_s*.png",
        em_glob: str = "*_s*.png",
    ) -> "GroundTruth":
        md = parse_metadata(metadata_path)
        gt = cls(metadata=md, mask_dir=Path(mask_dir),
                 em_dir=Path(em_dir) if em_dir else None, downscale=int(downscale))
        gt._mask_files = gt._index_slices(gt.mask_dir, mask_glob)
        if gt.em_dir:
            gt._em_files = gt._index_slices(gt.em_dir, em_glob)
        # neuron-label -> [nr,...] (a neuron can be split into several segments)
        by: Dict[str, List[int]] = {}
        for nr, lab in zip(md["nr"], md["label"]):
            by.setdefault(str(lab), []).append(int(nr))
        gt._by_label = by
        return gt

    @classmethod
    def from_config(cls, **kw) -> "GroundTruth":
        """Build from the GT_* paths in ``sam2_utils.config`` (the GT lives on the
        stable F: drive; the old local-copy shadowing was retired)."""
        if config is None:
            raise RuntimeError("sam2_utils.config not importable; use GroundTruth.load(...)")
        return cls.load(config.GT_METADATA, config.GT_MASK_DIR, config.GT_EM_DIR,
                        downscale=getattr(config, "GT_DOWNSCALE", 4), **kw)

    @staticmethod
    def _index_slices(folder: Path, glob: str) -> "OrderedDict[int, Path]":
        out: Dict[int, Path] = {}
        for p in folder.glob(glob):
            m = _SLICE_IDX.search(p.name)
            if m:
                out[int(m.group(1))] = p
        return OrderedDict(sorted(out.items()))

    # -- slice indexing --------------------------------------------------------
    @property
    def slice_indices(self) -> List[int]:
        return list(self._mask_files.keys())

    @property
    def n_slices(self) -> int:
        return len(self._mask_files)

    def has_slice(self, idx: int) -> bool:
        return int(idx) in self._mask_files

    # -- segment / neuron lookup ----------------------------------------------
    def nr_for_label(self, label: str) -> List[int]:
        """Segment numbers whose parsed neuron identity == ``label`` (exact)."""
        return list(self._by_label.get(str(label), []))

    def find(self, query: str) -> pd.DataFrame:
        """Metadata rows whose label or raw name contains ``query`` (case-insensitive)."""
        q = query.lower()
        md = self.metadata
        hit = (md["label"].str.lower().str.contains(q, regex=False)
               | md["name"].str.lower().str.contains(q, regex=False))
        return md[hit]

    def neurons(self, kind: Optional[str] = "neuron") -> List[str]:
        """Sorted unique labels, optionally filtered to one ``kind``."""
        md = self.metadata
        if kind is not None:
            md = md[md["kind"] == kind]
        return sorted(md["label"].unique().tolist())

    # -- pixel data ------------------------------------------------------------
    def label_slice(self, idx: int) -> np.ndarray:
        """The raw ``uint16`` labelmap for VAST slice ``idx`` (pixel == Nr)."""
        idx = int(idx)
        if idx in self._label_cache:
            self._label_cache.move_to_end(idx)
            return self._label_cache[idx]
        if idx not in self._mask_files:
            raise KeyError(f"no GT mask for slice {idx}; have {self.n_slices} slices")
        arr = _read_label_png(self._mask_files[idx])
        self._label_cache[idx] = arr
        self._label_cache.move_to_end(idx)
        while len(self._label_cache) > self._cache_cap:
            self._label_cache.popitem(last=False)
        return arr

    def em_slice(self, idx: int) -> np.ndarray:
        """The matching EM grayscale slice (requires ``em_dir``)."""
        idx = int(idx)
        if not self._em_files:
            raise RuntimeError("no em_dir configured for this GroundTruth")
        if idx not in self._em_files:
            raise KeyError(f"no EM slice {idx}")
        return np.asarray(Image.open(self._em_files[idx]))

    def segment_mask(self, idx: int, nr: int) -> np.ndarray:
        """Boolean mask of segment ``nr`` on slice ``idx``."""
        return self.label_slice(idx) == int(nr)

    def neuron_mask(self, idx: int, label: str) -> np.ndarray:
        """Boolean mask for a neuron label on slice ``idx`` (union of its segments).

        A neuron split into several VAST segments (fragments) is unioned. Raises
        if the label is unknown so a typo never silently scores as all-empty.
        """
        nrs = self.nr_for_label(label)
        if not nrs:
            raise KeyError(f"no GT segment with label {label!r}")
        lab = self.label_slice(idx)
        return np.isin(lab, np.asarray(nrs, dtype=lab.dtype))

    def present_segments(self, idx: int) -> Dict[int, int]:
        """``{nr: pixel_count}`` for every non-zero segment on slice ``idx``."""
        vals, counts = np.unique(self.label_slice(idx), return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts) if v != 0}

    def slices_with_segment(self, nr: int) -> List[int]:
        """VAST slice indices on which segment ``nr`` appears at least once.

        Reads every labelmap once — O(n_slices) IO. Use the metadata bbox z-range
        (:meth:`bbox_z_range`) first to bound the scan when you can.
        """
        nr = int(nr)
        return [i for i in self.slice_indices if bool((self.label_slice(i) == nr).any())]

    # -- metadata geometry -----------------------------------------------------
    def bbox_z_range(self, nr: int) -> Tuple[int, int]:
        """``(z1, z2)`` full-res VAST z-extent of segment ``nr`` (inclusive)."""
        r = self.metadata.loc[int(nr)]
        return int(r["bbox_z1"]), int(r["bbox_z2"])

    def bbox_in_mask_px(self, nr: int) -> Tuple[int, int, int, int]:
        """Segment ``nr`` xy bbox in **mask pixel** coords: ``(x1, y1, x2, y2)``.

        The metadata bbox is full-res VAST px; divide by :attr:`downscale` to land
        in the exported mask/EM grid.
        """
        r = self.metadata.loc[int(nr)]
        s = self.downscale
        return (int(r["bbox_x1"]) // s, int(r["bbox_y1"]) // s,
                int(r["bbox_x2"]) // s, int(r["bbox_y2"]) // s)
