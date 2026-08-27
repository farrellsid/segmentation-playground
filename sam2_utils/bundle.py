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
import shutil
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

#: state.json fields a recrop changes, and the only ones import merges back. The whole
#: `config` block is deliberately absent: it carries output_root and frames_root, which
#: are the reviewer's machine's paths, not the master tree's. An allowlist rather than a
#: denylist, so a field added to ChainState later cannot quietly start crossing machines.
#:
#: `prompts` is here because it is crop-space too, not just the frame/window fields:
#: pipeline/orchestrator.py's tier-2 anchor phase seeds state.prompts with the _pcrop
#: points/box for the window that chain ran in and never maps them back to _sam (see
#: the comment at orchestrator.py's box-seeding step). After a recrop the master's OLD
#: prompts sit next to the bundle's NEW crop_window, so opening the chain seeds a point
#: at the wrong offset and a re-predict runs from a positive point no longer on the
#: cell. The bundle's own prompts are already correct for the new window, since
#: run_chain re-seeds the anchor as part of the recrop.
GEOMETRY_FIELDS = ("crop_window", "frame_to_z", "n_frames", "anchor_frame_idx", "prompts")

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


def is_bundle(root) -> bool:
    """Whether ``root`` is a review bundle rather than a plain output tree.

    The manifest is the marker. Callers used to spell this inline, and the two
    behaviours that depend on it (frames stay relative, geometry comes home) are too
    important to be decided by a copy of an expression.
    """
    return (Path(root) / BUNDLE_MANIFEST).exists()


def adopt_chain_frames(chain_dir, frames_dir, *, relative: str = "frames") -> str:
    """Move a freshly prepared frame view into a bundle chain dir, returning its
    relative name.

    A recrop prepares frames in the machine's frames cache and records an absolute
    path. Inside a bundle that breaks two things at once: :func:`validate_bundle`
    rejects an absolute ``frames_dir``, and the bundle stops opening on any other
    machine. Moving the view in and recording it relative keeps the bundle
    self-contained.

    Moves rather than copies: the source is a cache ``prepare_chain_crop_frames``
    rebuilds fresh anyway, and a duplicated tier-2 chain's frames are real disk on a
    laptop.

    Parameters
    ----------
    chain_dir : Path
        The bundle's chain directory.
    frames_dir : path-like
        The prepared view to adopt.
    relative : str, optional
        The name to give it inside the chain directory.

    Returns
    -------
    str
        ``relative``, ready to store as the state's ``frames_dir``.

    Raises
    ------
    ValueError
        If the source holds no prepared frames. Replacing a chain's frames with an
        empty directory would leave the bundle unopenable, which is worse than a
        failed recrop.
    RuntimeError
        If ``dest`` cannot be cleared before the move. Something still has it open
        (napari's image layer keeps a handle on the OLD frames/ until the session
        closes it) or it is a symlink (``shutil.rmtree`` refuses to remove one).
        Moving onto a directory that is still there would nest the new view INSIDE
        it (``dest/<old view name>/``) rather than replace it, and everything
        downstream, including ``validate_bundle``, would find jpgs and pass, while
        the GUI draws the new masks on the old window's pixels.
    """
    chain_dir, frames_dir = Path(chain_dir), Path(frames_dir)
    if not sorted(frames_dir.glob("*.jpg")):
        raise ValueError(f"{frames_dir} holds no prepared frames; refusing to replace "
                         f"{chain_dir / relative} with an empty view")
    dest = chain_dir / relative
    if dest.exists() and frames_dir.resolve() == dest.resolve():
        return relative
    if dest.exists():
        # ignore_errors=True is deliberate: it means "don't raise for a directory
        # that partly does not exist", not "pretend the delete worked". Checking the
        # result afterward, rather than trusting shutil.rmtree's silence, is what
        # turns a swallowed failure into a refusal instead of a silent nest.
        shutil.rmtree(dest, ignore_errors=True)
        if dest.exists():
            raise RuntimeError(
                f"could not remove {dest} to adopt the recrop's frames; something "
                f"still has it open (close any viewer showing this chain first) or "
                f"it is a symlink. The new frames are still at {frames_dir}; "
                f"nothing was moved, so the chain keeps its OLD frames rather than "
                f"a silently nested mix of the two.")
    shutil.move(str(frames_dir), str(dest))
    return relative


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


def geometry_of(state: dict) -> dict:
    """The recrop-sensitive slice of a parsed ``state.json``.

    Comparing this, rather than the whole file, is what separates "she recropped this
    chain" from "these two files were written on different machines": ``frames_dir``,
    ``config`` and the timing blocks differ on every machine and mean nothing here.
    """
    return {k: state.get(k) for k in GEOMETRY_FIELDS}


def merge_geometry(master_state: dict, bundle_state: dict) -> dict:
    """``master_state`` with the bundle's geometry merged in, as a new dict.

    Clears ``frames_dir``, which is the part that is easy to miss.
    ``prepare_chain_crop_frames`` namespaces its view directory by neuron, chain and
    crop scale, so a new window at the SAME scale reuses the same directory name: the
    master's recorded path still exists and still holds the old window's frames, and
    the GUI would draw the new masks on them with nothing missing to signal it. None
    makes the next open regenerate from the raw EM, which the master machine has.
    """
    merged = dict(master_state)
    merged.update(geometry_of(bundle_state))
    merged["frames_dir"] = None
    return merged


def validate_bundle(bundle_root: Path, *, require_frames: bool = True) -> List[str]:
    """Check a bundle and return a list of problems, empty when it is well formed.

    Parameters
    ----------
    bundle_root : Path
        Directory holding ``bundle.json``.
    require_frames : bool, optional
        Whether every chain must have its frames on disk. True for a bundle being
        delivered: without frames there is nothing to draw on, and the reviewer
        finds out only when a chain fails to open.

        Pass False where their absence is expected and harmless. Two callers do.
        A git clone of a bundle carries masks and metadata only, because frames are
        static, roughly 200 times the size of everything else, and arrive once by
        drive instead. And ``import_bundle`` moves only masks and qc.csv, so it has
        no business demanding frames it will never read.

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
        state = json.loads(state_path.read_text(encoding="utf-8"))
        frames_dir = state.get("frames_dir")
        if frames_dir and is_recorded_absolute(frames_dir):
            problems.append(f"{entry['chain_dir']}: frames_dir is absolute ({frames_dir}); "
                            f"a bundle must record it relative or it will not open elsewhere")
        if require_frames:
            # An existing but empty directory counts as missing: that is the
            # half-finished copy, which looks fine until a chain will not open.
            here = chain_dir / (frames_dir or "frames")
            n_here = len(list(here.glob("*.jpg"))) if here.is_dir() else 0
            if not n_here:
                problems.append(f"{entry['chain_dir']}: no frames on disk; there is nothing to "
                                f"draw on. If this is a git clone, the frames arrive separately "
                                f"from the drive, so place them or validate with "
                                f"require_frames=False")

    # The manifest is the bundle's declaration of what it holds, but import walks
    # the DISK. A chain dir the manifest does not name used to validate clean and
    # then die inside import with a misleading "not exported from this tree", so
    # it is reported here, where the reason is visible.
    declared = {entry["chain_dir"] for entry in manifest.get("chains", [])}
    for rec in index_chains(bundle_root):
        if rec["chain_dir"] not in declared:
            problems.append(f"{rec['chain_dir']}: on disk but not listed in {BUNDLE_MANIFEST}; "
                            f"the manifest is the bundle's index and an import walks the disk, "
                            f"so the two must agree")
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

    Notes
    -----
    Only rows whose ``(neuron, chain_idx)`` is actually a chain here are counted.
    ``_review.csv`` outlives the chains it describes: a tree the ledger was copied
    into, or a bundle exporting a subset of the neurons the ledger covers, leaves
    rows for chains that are not on disk. Counting those produced totals like
    ``{"total": 1, "reviewed": 3}``, which the launcher displayed as "3/1
    reviewed". A row for a chain that is not here is also counted at most once,
    since the key set makes repeats collapse.
    """
    output_root = Path(output_root)
    progress: Dict[str, dict] = {}
    present = set()
    for rec in index_chains(output_root):
        progress.setdefault(rec["cell_name"], {"total": 0, "reviewed": 0})["total"] += 1
        present.add((rec["cell_name"], rec["chain_idx"]))

    review_path = output_root / "_review.csv"
    if review_path.exists():
        counted = set()
        with review_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                status = (row.get("review_status") or "").strip()
                name = row.get("neuron")
                try:
                    key = (name, int(row.get("chain_idx")))
                except (TypeError, ValueError):
                    continue
                if key in present and key not in counted and status and status != "pending":
                    counted.add(key)
                    progress[name]["reviewed"] += 1
    return progress
