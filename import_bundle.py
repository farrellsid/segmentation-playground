"""Merge a returned review bundle back into the master tree.

A bundle a reviewer sends back is the deliverable, so this is the half that makes
the round trip real. Only reviewer-owned content moves: the masks she redrew and
the per-chain qc.csv. Everything else in a bundle is a copy of state the master
tree already owns, and state.json in particular carries a RELATIVE frames_dir
that would break the master tree if copied over it.

    py -3 import_bundle.py --bundle "D:\\returned\\AIA_from_lucinda" \\
        --output-root "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed"
    py -3 import_bundle.py --bundle ... --output-root ... --dry-run
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path
from typing import List

from sam2_utils import bundle, chain_meta


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

    verb = "would update" if dry_run else "updated"
    print(f"[import] {verb} {len(changed)} chain(s)")
    for rec in changed:
        print(f"  {rec['chain_dir']}")
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
