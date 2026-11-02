"""Copy a bundle's EM frames from a drive into a git clone that has none.

A review bundle splits cleanly in two. The masks and metadata are about 9 MB and
change constantly, so they live in git. The frames are about 1.9 GB, never change,
and are regenerable from the raw EM, so they travel once on a drive and are
deliberately gitignored.

That leaves one manual step between cloning and reviewing: putting the frames
where the clone expects them. This does it, matching chain directories by name so
the two halves line up, and reporting any chain the drive turns out not to have
rather than leaving a hole to be discovered mid-review.

    py -3 place_frames.py --clone ~/mask-review/AIY_for_lucinda \\
        --from /Volumes/Expansion/Lucinda_Review/bundles/AIY_for_lucinda

Safe to re-run: a chain that already has its frames is skipped, so an interrupted
copy resumes where it stopped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

from sam2_utils import bundle


def place_frames(clone_root: Path, source_root: Path, *, dry_run: bool = False) -> Dict:
    """Copy each chain's frames from ``source_root`` into ``clone_root``.

    Parameters
    ----------
    clone_root : Path
        A cloned bundle: masks and metadata present, frames absent.
    source_root : Path
        The matching bundle on the drive, which has the frames.
    dry_run : bool, optional
        Report what would be copied without writing.

    Returns
    -------
    dict
        ``placed`` (chains copied this run), ``already`` (chains that already had
        their frames), ``missing`` (chain dirs the source does not have frames for,
        as POSIX strings), and ``frames`` (files copied).

    Raises
    ------
    SystemExit
        If either side is not a bundle. Copying between mismatched directories
        would scatter frames into places nothing reads them from.
    """
    clone_root, source_root = Path(clone_root), Path(source_root)
    for label, root in (("clone", clone_root), ("source", source_root)):
        if not (root / bundle.BUNDLE_MANIFEST).exists():
            raise SystemExit(f"{root} is not a bundle ({label}): no {bundle.BUNDLE_MANIFEST}")

    result: Dict = {"placed": 0, "already": 0, "missing": [], "frames": 0}
    for rec in bundle.index_chains(clone_root):
        rel = rec["chain_dir"]
        dst = clone_root / rel / (rec["state"].get("frames_dir") or "frames")
        src = source_root / rel / "frames"

        if dst.is_dir() and any(dst.glob("*.jpg")):
            result["already"] += 1
            continue
        if not src.is_dir() or not any(src.glob("*.jpg")):
            result["missing"].append(rel)
            continue

        n = len(list(src.glob("*.jpg")))
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
        result["placed"] += 1
        result["frames"] += n
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", type=Path, required=True,
                    help="the cloned bundle, which has masks but no frames")
    ap.add_argument("--from", dest="source", type=Path, required=True,
                    help="the matching bundle on the drive, which has the frames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    r = place_frames(args.clone, args.source, dry_run=args.dry_run)
    verb = "would place" if args.dry_run else "placed"
    print(f"[frames] {verb} {r['placed']} chain(s), {r['frames']} frame(s); "
          f"{r['already']} already had them")
    if r["missing"]:
        print(f"[frames] {len(r['missing'])} chain(s) have no frames on the source:",
              file=sys.stderr)
        for rel in r["missing"]:
            print(f"  {rel}", file=sys.stderr)
        print("[frames] those chains will not open until their frames are placed.",
              file=sys.stderr)
        raise SystemExit(1)

    problems = bundle.validate_bundle(args.clone)
    if problems and not args.dry_run:
        print("[frames] the clone still has problems:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(1)
    if not args.dry_run:
        print("[frames] clone validates clean and is ready to review.")


if __name__ == "__main__":
    main()
