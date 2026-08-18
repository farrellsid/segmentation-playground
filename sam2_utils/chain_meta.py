"""meta.json: a chain's portable identity and geometry record.

An exporter needs to know which neuron a mask belongs to and where its pixels sit
in the full-resolution frame. Both facts exist today only inside ``state.json``
and the directory name, and reading ``state.json`` properly means importing
``pipeline``. A Blender or VAST exporter that imports ``pipeline`` breaks every
time the pipeline is refactored, so this module defines a small, versioned record
with a stable schema and reads nothing but plain JSON.

``state.json`` stays authoritative for anything the pipeline itself reads back.
This is a projection of it, not a competing source of truth.

Deliberately imports no ``pipeline`` (that would also be circular, since
``pipeline`` imports ``sam2_utils``), no torch, and no cv2.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

#: Bump when the schema changes incompatibly. Readers refuse anything newer.
META_SCHEMA_VERSION = 1

#: Filename written inside a chain directory.
META_FILENAME = "meta.json"

_REQUIRED = ("schema_version", "neuron_id", "cell_name", "chain_idx",
             "mask_space", "mask_scale", "crop_window", "z_range", "provenance")


def _z_range_from_masks(chain_dir: Optional[Path]):
    """Return ``[min_z, max_z]`` over ``masks/mask_<z>.png``, or None if absent."""
    if chain_dir is None:
        return None
    masks = Path(chain_dir) / "masks"
    if not masks.is_dir():
        return None
    zs = []
    for p in masks.glob("mask_*.png"):
        stem = p.stem.split("_")[-1]
        if stem.isdigit():
            zs.append(int(stem))
    return [min(zs), max(zs)] if zs else None


def build_meta(state: dict, *, neuron_id: int, source_tree: str, backend: str = "sam2",
               reprop_variant: Optional[str] = None, exported: Optional[str] = None,
               chain_dir: Optional[Path] = None) -> dict:
    """Build a chain's metadata record from its loaded ``state.json`` dict.

    Parameters
    ----------
    state : dict
        The chain's ``state.json``, already parsed. A plain dict, not a
        ``ChainState``, so this module stays free of ``pipeline``.
    neuron_id : int
        The permanent id from :mod:`sam2_utils.registry`.
    source_tree : str
        Name of the output tree this chain came from, for provenance.
    backend : str, optional
        ``"sam2"`` or ``"sam3"``.
    reprop_variant : str, optional
        ``"mask_seed"`` or ``"box_seed"`` when this chain came from a
        re-propagation run, else None.
    exported : str, optional
        ISO timestamp. Defaults to now, in UTC.
    chain_dir : Path, optional
        The chain directory, read only to derive ``z_range`` from the mask files.

    Returns
    -------
    dict
        The metadata record. ``mask_scale`` is per chain on purpose: a legacy
        ``_sam`` chain carries the pipeline ``save_downscale`` (8), a tier-2
        ``_pcrop`` chain carries the crop's own ``crop_scale`` (often 2, coarser
        when ``chain_crop_max_px`` forced the read down). An exporter that assumed
        one global mask scale would misplace every tier-2 chain.
    """
    cw = state.get("crop_window")
    if cw:
        mask_space = "_pcrop"
        mask_scale = int(cw.get("crop_scale", state.get("save_downscale", 8)))
    else:
        mask_space = "_sam"
        mask_scale = int(state.get("save_downscale", 8))
    return {
        "schema_version": META_SCHEMA_VERSION,
        "neuron_id": int(neuron_id),
        "cell_name": state["neuron"],
        "chain_idx": int(state["chain_idx"]),
        "mask_space": mask_space,
        "crop_window": cw,
        "z_range": _z_range_from_masks(chain_dir),
        "mask_scale": mask_scale,
        "provenance": {
            "source_tree": source_tree,
            "backend": backend,
            "reprop_variant": reprop_variant,
            "exported": exported or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def validate_meta(meta: dict) -> None:
    """Raise ``ValueError`` if ``meta`` is missing a field or is too new to read.

    Parameters
    ----------
    meta : dict
        A record from :func:`build_meta` or :func:`read_meta`.
    """
    missing = [k for k in _REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"meta.json missing required field(s): {', '.join(missing)}")
    version = int(meta["schema_version"])
    if version > META_SCHEMA_VERSION:
        raise ValueError(f"meta.json schema_version {version} is newer than this code "
                         f"supports ({META_SCHEMA_VERSION}); update the repo before reading it")


def write_meta(chain_dir: Path, meta: dict) -> Path:
    """Validate ``meta`` and write it to ``<chain_dir>/meta.json``.

    Returns
    -------
    Path
        The written path.
    """
    validate_meta(meta)
    path = Path(chain_dir) / META_FILENAME
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_meta(chain_dir: Path) -> dict:
    """Read and validate ``<chain_dir>/meta.json``.

    Returns
    -------
    dict
        The metadata record.
    """
    path = Path(chain_dir) / META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    validate_meta(meta)
    return meta
