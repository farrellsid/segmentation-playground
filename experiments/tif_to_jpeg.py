"""Convert a directory of EM .tif slices to JPEG. Standalone and reusable.

Only needs OpenCV (numpy comes with it); no repo imports, so it runs anywhere:

    py -3 tif_to_jpeg.py --in  "F:\\ZhenLab\\Data\\SAM2_test_NR_raw" \\
                         --out "G:\\farrell - temp for jpeg stuff" \\
                         --quality 90

Each input ``<name>.tif`` becomes ``<name>.jpg`` in the output dir, so the z encoded
in the Zhen filenames (e.g. ``1301____z1300.0.tif``) is preserved. EM slices are 8-bit
grayscale, so the JPEG is written single-channel (smaller, no colour is invented).

Options:
  --quality N   JPEG quality 1..100 (default 90). EM is noise-heavy, so JPEG does not
                shrink it dramatically: at full res expect roughly 30 MB/slice at q85,
                36 at q90, 48 at q95.
  --scale  S    Integer area-downscale before encoding (default 1 = full resolution).
                S=2 gives half resolution, and so on.
  --overwrite   Re-encode even if the .jpg already exists (default: skip, so a stopped
                run resumes where it left off).

The run prints a per-file line only on error; otherwise a progress bar (tqdm if present,
else a plain counter) and a final summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def convert_dir(in_dir: Path, out_dir: Path, quality: int, scale: int,
                overwrite: bool, pattern: str) -> None:
    tifs = sorted(in_dir.glob(pattern))
    if not tifs:
        print(f"[tif2jpg] no files matching {pattern!r} in {in_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    # progress bar if tqdm is around, else a lightweight fallback
    try:
        from tqdm import tqdm
        it = tqdm(tifs, unit="slice", desc=f"tif -> jpg q{quality} s{scale}")
    except Exception:
        it = tifs

    done = skipped = failed = 0
    total_bytes = 0
    t0 = time.time()
    for i, tif in enumerate(it):
        dst = out_dir / (tif.stem + ".jpg")
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        img = cv2.imread(str(tif), cv2.IMREAD_GRAYSCALE)  # 8-bit grayscale EM
        if img is None:
            print(f"[tif2jpg] FAILED to read {tif.name}")
            failed += 1
            continue
        if scale > 1:
            img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale,
                             interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(dst), img, params):
            print(f"[tif2jpg] FAILED to write {dst.name}")
            failed += 1
            continue
        done += 1
        total_bytes += dst.stat().st_size
        if it is tifs and (i + 1) % 20 == 0:  # plain-counter fallback progress
            print(f"[tif2jpg] {i + 1}/{len(tifs)}")

    el = time.time() - t0
    gb = total_bytes / 1e9
    print(f"[tif2jpg] done: {done} converted, {skipped} skipped, {failed} failed "
          f"in {el:.0f}s. Wrote {gb:.1f} GB to {out_dir}"
          + (f" ({gb / done * 1000:.1f} MB/slice avg)" if done else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="input dir of .tif slices")
    ap.add_argument("--out", dest="out_dir", required=True, help="output dir for .jpg")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1..100 (default 90)")
    ap.add_argument("--scale", type=int, default=1, help="integer downscale (1 = full res)")
    ap.add_argument("--pattern", default="*.tif", help="glob for input files (default *.tif)")
    ap.add_argument("--overwrite", action="store_true", help="re-encode existing .jpg")
    args = ap.parse_args(argv)
    convert_dir(Path(args.in_dir), Path(args.out_dir), args.quality, args.scale,
                args.overwrite, args.pattern)


if __name__ == "__main__":
    sys.exit(main())
