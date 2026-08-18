"""Review bundle logic: a self-contained directory a second reviewer can open.

A bundle is an output tree stripped to what a reviewer needs and made portable:
identity from meta.json, masks, per-chain QC, the frames to draw on, and a
manifest. Its ``state.json`` files carry a RELATIVE ``frames_dir``, which is what
lets the bundle open on a machine with none of the original paths and no raw EM
store to regenerate frames from.

Pure logic only: no file copying (that is export_bundle.py), no pipeline import
(the spec requires bundle contents be readable without it, and pipeline imports
sam2_utils, so importing back would be circular), no torch, no cv2.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

#: Bump when the bundle layout changes incompatibly. Readers refuse anything newer.
BUNDLE_SCHEMA_VERSION = 1

#: Manifest filename at the bundle root.
BUNDLE_MANIFEST = "bundle.json"

#: What a reviewer can change, and therefore all that import_bundle merges back.
#: Everything else in a bundle is a copy of state the master tree already owns.
REVIEWER_OWNED = ("masks", "qc.csv")

#: Tree-root ledgers the review GUI appends to, merged back ROW-WISE (not copied)
#: by import_bundle. Separate from REVIEWER_OWNED, which is the per-chain set: a
#: master tree's copy of these holds rows for chains that were never in the
#: bundle, and copying the file over would delete them.
REVIEWER_LEDGERS = ("_review.csv", "_labels.csv")

#: Directory inside a bundle holding the CATMAID slice the GUI reads.
BUNDLE_DATA_DIR = "data"

#: Bundle-local chain list: the subset of ``data/chains.json`` this bundle needs.
BUNDLE_CHAINS_NAME = "chains.json"

#: Bundle-local node table: the subset of ``data/aggregate_data_pv.csv`` this
#: bundle needs. Named for its role rather than carrying the source filename
#: forward, because it is a subset, not a copy of that file.
BUNDLE_NODES_NAME = "nodes.csv"

#: Both of the above, relative to the bundle root. A bundle without them cannot
#: be opened: both source files are gitignored, so a reviewer who clones the repo
#: has neither, and ``ReviewContext`` would raise ``FileNotFoundError`` on the
#: first chain she opens.
BUNDLE_DATA_FILES = (f"{BUNDLE_DATA_DIR}/{BUNDLE_CHAINS_NAME}",
                     f"{BUNDLE_DATA_DIR}/{BUNDLE_NODES_NAME}")


def is_recorded_absolute(recorded) -> bool:
    """Whether a recorded ``frames_dir`` should be treated as an absolute path.

    Parameters
    ----------
    recorded : str or Path
        The ``frames_dir`` value out of a ``state.json``.

    Returns
    -------
    bool
        True when the value is absolute.

    Notes
    -----
    A leading ``/`` (a Narval scratch path such as ``/localscratch/<jobid>/...``)
    or a leading backslash counts as absolute even on Windows, where ``Path`` would
    otherwise read a leading slash as "root of the current drive" and ``is_absolute`` would
    return False. Treating such a value as relative would silently join a dead
    cluster path onto a chain directory instead of failing where it is
    recognisable.
    """
    text = str(recorded)
    return text.startswith(("/", "\\")) or Path(text).is_absolute()


def resolve_frames_dir(recorded, chain_dir):
    """Resolve a chain's recorded ``frames_dir`` to a usable path.

    THE one implementation of this rule. ``gui.resolve_frames_dir`` delegates
    here, :func:`validate_bundle` uses :func:`is_recorded_absolute`, and
    ``export_bundle`` resolves its copy source through it, so the three callers
    cannot drift apart again. They did: export used to test only a leading ``/``
    and then call a bare ``Path(recorded)``, which resolved a RELATIVE
    ``frames_dir`` (exactly what a bundle records) against the current working
    directory, and a re-export from a bundle happily copied a decoy ``frames/``
    out of the cwd.

    Parameters
    ----------
    recorded : str, Path or None
        The ``frames_dir`` value from a ``state.json``.
    chain_dir : Path
        The chain directory the state was loaded from.

    Returns
    -------
    Path or None
        ``chain_dir / recorded`` when recorded is relative, the path unchanged
        when it is absolute, and None when nothing is recorded.
    """
    if recorded is None:
        return None
    text = str(recorded)
    if is_recorded_absolute(text):
        return Path(text)
    return Path(chain_dir) / text


def index_chains(output_root: Path, neurons: Optional[List[str]] = None) -> List[dict]:
    """Index every chain under ``output_root``.

    Parameters
    ----------
    output_root : Path
        A tree laid out as ``<neuron>/chain_<idx>/state.json``.
    neurons : list of str, optional
        Restrict to these cell names. None includes everything.

    Returns
    -------
    list of dict
        One record per chain: ``cell_name``, ``chain_idx``, ``chain_dir`` (relative
        to ``output_root``, as a POSIX string so a manifest written on Windows reads
        on a Mac), and ``state`` (the parsed state.json).
    """
    output_root = Path(output_root)
    wanted = set(neurons) if neurons else None
    out: List[dict] = []
    for state_path in sorted(output_root.glob("*/chain_*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        name = state.get("neuron", state_path.parent.parent.name)
        if wanted is not None and name not in wanted:
            continue
        out.append({
            "cell_name": name,
            "chain_idx": int(state.get("chain_idx", int(state_path.parent.name.split("_")[-1]))),
            "chain_dir": state_path.parent.relative_to(output_root).as_posix(),
            "state": state,
        })
    return out


def build_manifest(chains: List[dict], *, source_tree: str) -> dict:
    """Build the bundle manifest from an index.

    Parameters
    ----------
    chains : list of dict
        Records from :func:`index_chains`.
    source_tree : str
        The tree the bundle was exported from, for provenance.

    Returns
    -------
    dict
        The manifest. The heavy ``state`` blob is dropped: the manifest is an index,
        and each chain's own state.json remains authoritative.
    """
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_tree": source_tree,
        "chains": [{k: c[k] for k in ("cell_name", "chain_idx", "chain_dir")} for c in chains],
    }


def rewrite_state_frames_dir(state: dict, relative: str = "frames") -> dict:
    """Return a copy of ``state`` whose ``frames_dir`` is relative.

    Parameters
    ----------
    state : dict
        A parsed state.json. Not mutated.
    relative : str, optional
        The path to record, relative to the chain directory.

    Returns
    -------
    dict
        A shallow copy with ``frames_dir`` replaced.
    """
    out = dict(state)
    out["frames_dir"] = relative
    return out


def validate_bundle(bundle_root: Path) -> List[str]:
    """Check a bundle and return a list of problems, empty when it is well formed.

    Parameters
    ----------
    bundle_root : Path
        Directory holding ``bundle.json``.

    Returns
    -------
    list of str
        Human-readable problems. Returning a list rather than raising lets a caller
        report everything wrong at once, which matters when the bundle has already
        been mailed to someone.
    """
    bundle_root = Path(bundle_root)
    problems: List[str] = []
    manifest_path = bundle_root / BUNDLE_MANIFEST
    if not manifest_path.exists():
        return [f"missing {BUNDLE_MANIFEST} at {bundle_root}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("schema_version", 0))
    if version > BUNDLE_SCHEMA_VERSION:
        problems.append(f"bundle schema_version {version} is newer than this code supports "
                        f"({BUNDLE_SCHEMA_VERSION}); update the repo before opening it")
        return problems

    for rel in BUNDLE_DATA_FILES:
        if not (bundle_root / rel).exists():
            problems.append(f"missing {rel}; without it the reviewer's GUI falls back to the "
                            f"config paths, which do not exist on her machine, and no chain "
                            f"will open")
    for entry in manifest.get("chains", []):
        chain_dir = bundle_root / entry["chain_dir"]
        if not chain_dir.is_dir():
            problems.append(f"missing chain directory {entry['chain_dir']}")
            continue
        if not (chain_dir / "meta.json").exists():
            problems.append(f"{entry['chain_dir']}: missing meta.json")
        state_path = chain_dir / "state.json"
        if not state_path.exists():
            problems.append(f"{entry['chain_dir']}: missing state.json")
            continue
        frames_dir = json.loads(state_path.read_text(encoding="utf-8")).get("frames_dir")
        if frames_dir and is_recorded_absolute(frames_dir):
            problems.append(f"{entry['chain_dir']}: frames_dir is absolute ({frames_dir}); "
                            f"a bundle must record it relative or it will not open elsewhere")
    return problems


def review_progress(output_root: Path) -> Dict[str, dict]:
    """Count chains and reviewed chains per neuron.

    Parameters
    ----------
    output_root : Path
        A tree or bundle root, optionally containing ``_review.csv``.

    Returns
    -------
    dict
        ``{cell_name: {"total": int, "reviewed": int}}``. A chain counts as reviewed
        when ``_review.csv`` gives it a ``review_status`` other than empty or
        ``"pending"``.
    """
    output_root = Path(output_root)
    progress: Dict[str, dict] = {}
    for rec in index_chains(output_root):
        progress.setdefault(rec["cell_name"], {"total": 0, "reviewed": 0})["total"] += 1

    review_path = output_root / "_review.csv"
    if review_path.exists():
        with review_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                status = (row.get("review_status") or "").strip()
                name = row.get("neuron")
                if name in progress and status and status != "pending":
                    progress[name]["reviewed"] += 1
    return progress
