"""Merge a returned review bundle back into the master tree.

A bundle a reviewer sends back is the deliverable, so this is the half that makes
the round trip real. Only reviewer-owned content moves: per chain, the masks she
redrew and the qc.csv (``bundle.REVIEWER_OWNED``); at the tree root, her rows in
``_review.csv`` and ``_labels.csv`` (``bundle.REVIEWER_LEDGERS``). Everything else
in a bundle is a copy of state the master tree already owns, and state.json in
particular carries a RELATIVE frames_dir that would break the master tree if
copied over it.

The two root ledgers are merged ROW-WISE, never copied. Inside a bundle
``output_root`` IS the bundle, so the GUI writes a reviewer's chain dispositions
and per-frame verdicts there, and four of the ten keys review mode exposes
(``w``, ``o``, ``a``, ``x``) write to nothing else. A master tree's copy of the
same ledgers holds rows for chains that were never in the bundle, so copying the
file over would delete other people's work.

    py -3 import_bundle.py --bundle "D:\\returned\\AIA_from_lucinda" \\
        --output-root "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed"
    py -3 import_bundle.py --bundle ... --output-root ... --dry-run
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from sam2_utils import bundle, chain_meta, labels as labels_mod, review_queue


def _differs(src: Path, dst: Path) -> bool:
    """True when ``src`` and ``dst`` differ, treating a missing ``dst`` as different.

    ``filecmp.dircmp``'s default ``shallow=True`` compares common files by
    ``os.stat`` (size and mtime) only, not content. Two independently written
    same-size files can land on the same mtime (coarse filesystem timestamp
    resolution, or just two writes close enough together), which would make a
    genuinely changed reviewer mask read as unchanged and get silently skipped.
    ``shallow=False`` forces a byte-for-byte comparison of the common files.

    Assumes a FLAT mask directory. ``dircmp`` does not recurse: only top-level
    ``diff_files`` is inspected and ``common_dirs`` is never examined. Every mask
    writer in this repo emits a flat directory of ``mask_<z>.png``, so that holds
    today. If a nested mask layout is ever introduced, a change inside a
    subdirectory would go undetected here and a reviewer's edit would be silently
    dropped, so this comparison must be revisited alongside any such change.
    """
    if not dst.exists():
        return True
    if src.is_dir():
        cmp = filecmp.dircmp(str(src), str(dst), shallow=False)
        return bool(cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files)
    return not filecmp.cmp(str(src), str(dst), shallow=False)


#: The bundle's two root ledgers, in merge order, for reporting.
_REVIEW_LEDGER, _LABELS_LEDGER = bundle.REVIEWER_LEDGERS


def _manifest_keys(bundle_root: Path) -> Set[Tuple[str, int]]:
    """The ``(cell_name, chain_idx)`` pairs the bundle's manifest declares.

    Parameters
    ----------
    bundle_root : Path
        A validated bundle.

    Returns
    -------
    set of (str, int)
        The bundle's chains. Ledger rows outside this set are not the reviewer's
        work on this bundle, and are left where they are.
    """
    manifest = json.loads((bundle_root / bundle.BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    return {(str(e["cell_name"]), int(e["chain_idx"])) for e in manifest.get("chains", [])}


def _read_ledger(path: Path, columns) -> pd.DataFrame:
    """Read a ledger CSV into its declared schema, or an empty frame of it."""
    if not Path(path).exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_csv(path).reindex(columns=list(columns))


def _keyed(df: pd.DataFrame, key_columns) -> pd.Series:
    """A row-wise tuple key over ``key_columns``.

    Normalised so a CSV round trip cannot make ``("AIAL", 0)`` and
    ``("AIAL", "0")`` look like different chains.
    """
    if not len(df):
        return pd.Series([], dtype=object)

    def norm(row):
        out = []
        for col in key_columns:
            value = row[col]
            out.append(str(value) if col == "neuron" else int(value))
        return tuple(out)

    return df.apply(norm, axis=1)


def _is_newer(candidate, existing) -> bool:
    """Whether ``candidate``'s ISO timestamp is strictly later than ``existing``'s.

    False whenever either side is missing or unparseable, so an incoming row with
    a broken ``ts`` can never displace a master row that has a good one.
    """
    try:
        left = pd.Timestamp(candidate)
        right = pd.Timestamp(existing)
    except (ValueError, TypeError):
        return False
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left > right)


def merge_review_ledger(bundle_root: Path, output_root: Path,
                        chain_keys: Set[Tuple[str, int]], *,
                        dry_run: bool = False) -> Dict[str, int]:
    """Merge the bundle's ``_review.csv`` rows into the master's, by chain.

    Parameters
    ----------
    bundle_root, output_root : Path
        The returned bundle and the master tree.
    chain_keys : set of (str, int)
        The bundle's chains, from its manifest.
    dry_run : bool, optional
        Count what would change without writing.

    Returns
    -------
    dict
        ``{"updated": int, "appended": int}``. Both zero, and nothing written,
        when the bundle has no ``_review.csv`` (a reviewer who recorded no
        dispositions).

    Notes
    -----
    ``_review.csv`` holds exactly one row per ``(neuron, chain_idx)``
    (``review_queue.KEY_COLUMNS``), which the GUI upserts. So the reviewer's row
    for a chain in the bundle REPLACES the master's row for that chain, and every
    other master row, including chains that were never in the bundle, is left
    untouched. The file is written by ``review_queue``'s own atomic writer, since
    that module owns this ledger.
    """
    src = Path(bundle_root) / _REVIEW_LEDGER
    counts = {"updated": 0, "appended": 0}
    if not src.exists():
        return counts

    columns = review_queue.REVIEW_COLUMNS
    incoming = _read_ledger(src, columns)
    if len(incoming):
        incoming = incoming[_keyed(incoming, review_queue.KEY_COLUMNS).isin(chain_keys)]
    if not len(incoming):
        return counts

    master_path = Path(output_root) / _REVIEW_LEDGER
    master = _read_ledger(master_path, columns)
    master_keys = set(_keyed(master, review_queue.KEY_COLUMNS)) if len(master) else set()

    incoming_keys = _keyed(incoming, review_queue.KEY_COLUMNS)
    counts["updated"] = int(sum(1 for k in incoming_keys if k in master_keys))
    counts["appended"] = int(len(incoming) - counts["updated"])
    if dry_run:
        return counts

    if len(master):
        master = master[~_keyed(master, review_queue.KEY_COLUMNS).isin(set(incoming_keys))]
    merged = pd.concat([master, incoming], ignore_index=True)[list(columns)]
    review_queue._atomic_write_csv(merged, master_path)
    return counts


def merge_labels_ledger(bundle_root: Path, output_root: Path,
                        chain_keys: Set[Tuple[str, int]], *,
                        dry_run: bool = False) -> Dict[str, int]:
    """Merge the bundle's ``_labels.csv`` rows into the master's, by frame.

    Parameters
    ----------
    bundle_root, output_root : Path
        The returned bundle and the master tree.
    chain_keys : set of (str, int)
        The bundle's chains, from its manifest.
    dry_run : bool, optional
        Count what would change without writing.

    Returns
    -------
    dict
        ``{"updated": int, "appended": int}``. Both zero, and nothing written,
        when the bundle has no ``_labels.csv``.

    Notes
    -----
    De-duplication is on ``labels.KEY_COLUMNS``, ``(neuron, chain_idx, z)``. That
    is the store's own primary key: ``LabelStore._append`` drops any existing row
    with the same triple before appending, so re-labelling a frame overwrites its
    prior row rather than piling up duplicates. Using the same key here keeps the
    merged file obeying the invariant the store maintains. The ``ts`` column is
    provenance, not identity: folding it into the key would turn every re-labelled
    frame into a second row and break that invariant.

    A key the master does not have is appended. A key both sides have is a
    genuine collision, two humans labelling the same frame, and the row with the
    later ``ts`` wins. A returned bundle's row is normally the later one, since
    the reviewer works after the export, and her verdict is the final word by the
    division of labour. An unparseable or missing ``ts`` on either side keeps the
    master's row, so a malformed incoming row can never silently displace a good
    one.
    """
    src = Path(bundle_root) / _LABELS_LEDGER
    counts = {"updated": 0, "appended": 0}
    if not src.exists():
        return counts

    columns = labels_mod.LABEL_COLS
    incoming = _read_ledger(src, columns)
    if len(incoming):
        incoming = incoming[_keyed(incoming, review_queue.KEY_COLUMNS).isin(chain_keys)]
    if not len(incoming):
        return counts

    master_path = Path(output_root) / _LABELS_LEDGER
    master = _read_ledger(master_path, columns)

    incoming = incoming.copy()
    incoming["_key"] = _keyed(incoming, labels_mod.KEY_COLUMNS)
    if len(master):
        master = master.copy()
        master["_key"] = _keyed(master, labels_mod.KEY_COLUMNS)
        master_ts = dict(zip(master["_key"], master["ts"]))
    else:
        master_ts = {}

    take = set()
    for _, row in incoming.iterrows():
        key = row["_key"]
        if key not in master_ts:
            take.add(key)
            counts["appended"] += 1
        elif _is_newer(row.get("ts"), master_ts[key]):
            take.add(key)
            counts["updated"] += 1
    if dry_run or not take:
        return counts

    incoming = incoming[incoming["_key"].isin(take)]
    if len(master):
        master = master[~master["_key"].isin(take)]
    merged = pd.concat([master, incoming], ignore_index=True)[list(columns)]
    labels_mod._atomic_write_csv(merged, master_path)
    return counts


def import_bundle(bundle_root: Path, output_root: Path, *, dry_run: bool = False) -> List[dict]:
    """Merge reviewer-owned files from ``bundle_root`` into ``output_root``.

    Parameters
    ----------
    bundle_root : Path
        A returned bundle. Validated before anything is written.
    output_root : Path
        The master tree to merge into.
    dry_run : bool, optional
        Report what would change without writing.

    Returns
    -------
    list of dict
        One record per changed chain: ``chain_dir``, ``cell_name``, ``chain_idx``.

    Raises
    ------
    SystemExit
        If the bundle fails validation, or names a chain the master tree does not
        have. Both mean the bundle and the tree do not belong together, and a
        partial merge would be worse than no merge.

    Notes
    -----
    The per-chain files move first, then the two root ledgers merge row-wise. A
    reviewer's verdict keys (``w``, ``o``, ``a``, ``x``) write to nothing else, so
    without the ledger merge four of the ten controls review mode gives her would
    produce work that never comes home.
    """
    bundle_root, output_root = Path(bundle_root), Path(output_root)

    problems = bundle.validate_bundle(bundle_root)
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(f"{bundle_root} is not a valid bundle; refusing to import")

    changed: List[dict] = []
    for rec in bundle.index_chains(bundle_root):
        src_dir = bundle_root / rec["chain_dir"]
        dst_dir = output_root / rec["chain_dir"]
        if not dst_dir.is_dir():
            raise SystemExit(f"{rec['chain_dir']} is in the bundle but not in {output_root}; "
                             f"this bundle was not exported from this tree")

        meta = chain_meta.read_meta(src_dir)
        if meta["cell_name"] != rec["cell_name"]:
            raise SystemExit(f"{rec['chain_dir']}: meta.json says {meta['cell_name']!r} but the "
                             f"bundle index says {rec['cell_name']!r}")

        chain_changed = False
        for name in bundle.REVIEWER_OWNED:
            src, dst = src_dir / name, dst_dir / name
            if not src.exists() or not _differs(src, dst):
                continue
            chain_changed = True
            if dry_run:
                continue
            if src.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        if chain_changed:
            changed.append({k: rec[k] for k in ("chain_dir", "cell_name", "chain_idx")})

    chain_keys = _manifest_keys(bundle_root)
    review_counts = merge_review_ledger(bundle_root, output_root, chain_keys, dry_run=dry_run)
    label_counts = merge_labels_ledger(bundle_root, output_root, chain_keys, dry_run=dry_run)

    verb = "would update" if dry_run else "updated"
    print(f"[import] {verb} {len(changed)} chain(s)")
    for rec in changed:
        print(f"  {rec['chain_dir']}")
    for name, counts in ((_REVIEW_LEDGER, review_counts), (_LABELS_LEDGER, label_counts)):
        print(f"[import] {name}: {verb} {counts['updated']} row(s), "
              f"appended {counts['appended']} row(s)")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import_bundle(args.bundle, args.output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
