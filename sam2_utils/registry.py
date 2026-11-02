"""The neuron identity registry: a permanent cell_name to neuron_id mapping.

Identity used to be a directory name, and numeric ids were assigned by
enumeration order at render time, so an id meant nothing outside the render that
produced it. This module makes identity data instead: ids are assigned once,
committed to ``data/neuron_registry.csv``, and never renumbered, so a segment
number in an export means the same neuron forever.

Keep this module import-light (no torch, no cv2, no pandas, no network) so it can
be imported from an exporter that has none of those installed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

#: Columns of data/neuron_registry.csv, in order.
REGISTRY_COLUMNS = ["neuron_id", "cell_name", "first_seen", "notes"]

#: The shipped registry, resolved relative to this file so it works from any cwd.
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "neuron_registry.csv"


class UnknownNeuronError(KeyError):
    """Raised when a cell name or neuron id is not in the registry."""


def load_registry(path: Optional[Path] = None) -> Dict[str, int]:
    """Load the registry as a ``cell_name`` to ``neuron_id`` mapping.

    Parameters
    ----------
    path : Path, optional
        Registry CSV to read. Defaults to the shipped ``REGISTRY_PATH``.

    Returns
    -------
    dict
        ``{cell_name: neuron_id}``.

    Raises
    ------
    ValueError
        If an id is not a positive integer (0 is reserved for background) or if
        an id or name is duplicated. Both mean the file is corrupt, and a corrupt
        registry silently mislabels every export, so this fails rather than warns.
    """
    path = Path(path) if path is not None else REGISTRY_PATH
    mapping: Dict[str, int] = {}
    seen_ids: Dict[int, str] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row["cell_name"] or "").strip()
            nid = int(row["neuron_id"])
            if nid < 1:
                raise ValueError(f"{path}: neuron_id must be >= 1 (0 is background), got {nid}")
            if nid in seen_ids:
                raise ValueError(f"{path}: duplicate neuron_id {nid} "
                                 f"({seen_ids[nid]!r} and {name!r})")
            if name in mapping:
                raise ValueError(f"{path}: duplicate cell_name {name!r}")
            seen_ids[nid] = name
            mapping[name] = nid
    return mapping


def neuron_id(cell_name: str, *, registry: Optional[Dict[str, int]] = None) -> int:
    """Return the permanent id for ``cell_name``.

    Parameters
    ----------
    cell_name : str
        The neuron's name as it appears in ``data/chains.json``.
    registry : dict, optional
        A preloaded mapping from :func:`load_registry`. Pass one in a loop to
        avoid re-reading the file per lookup.

    Returns
    -------
    int
        The neuron id, always >= 1.

    Raises
    ------
    UnknownNeuronError
        If the name is not registered. Adding a neuron is an explicit act (append
        a row), so an unknown name is a mistake rather than something to invent.
    """
    reg = load_registry() if registry is None else registry
    try:
        return reg[cell_name]
    except KeyError:
        raise UnknownNeuronError(cell_name) from None


def neuron_name(nid: int, *, registry: Optional[Dict[str, int]] = None) -> str:
    """Return the cell name for a neuron id, the inverse of :func:`neuron_id`.

    Parameters
    ----------
    nid : int
        A registered neuron id.
    registry : dict, optional
        A preloaded mapping from :func:`load_registry`.

    Returns
    -------
    str
        The cell name.

    Raises
    ------
    UnknownNeuronError
        If the id is not registered.
    """
    reg = load_registry() if registry is None else registry
    for name, value in reg.items():
        if value == nid:
            return name
    raise UnknownNeuronError(nid)
