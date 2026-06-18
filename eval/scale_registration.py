"""scale_registration.py — derive the full-res registration from the ¼-scale fit by ×4.

Stage 0.3 switched the GT to full_scale (GT_DOWNSCALE 4→1), which invalidates the
¼-scale registration.json (linear part ≈ 0.25·I). A from-scratch full-res re-fit is
correct but slow (851 × 89.6 MB HDD PNG decodes, ~1.5 h). It's also unnecessary: the
full-res registration is *geometrically identical* to the ¼ fit with every parameter
scaled ×4, because a full_scale pixel is exactly 4× a quarter pixel at the same
physical point:

    mask_full = 4·mask_quarter = 4·(skel @ L.T + t) = skel @ (4L).T + (4t)

so L→4L and t→4t for every section (and the mean A→4A, offsets→4·offsets). The only
thing a true re-fit adds is marginally finer sub-pixel centroids — negligible for
placing prompts on ~20–40 px neurites.

Run:  py -3 -u -m eval.scale_registration              # scale + light validate
      py -3 -u -m eval.scale_registration --no-validate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sam2_utils import config
from .registration import Registration, on_mask_rate
from .groundtruth import GroundTruth


def scale_registration(reg: Registration, factor: float) -> Registration:
    return Registration(
        A=reg.A * factor,
        offsets=reg.offsets * factor,
        z_min=reg.z_min,
        affines=None if reg.affines is None else reg.affines * factor,
    )


def main() -> None:
    base = config.DATA_DIR / "groundtruth" / "skeletons_p280"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=base / "registration_quarter_scale.json",
                    help="¼-scale fit to scale up")
    ap.add_argument("--out", type=Path, default=base / "registration.json")
    ap.add_argument("--skeleton-csv", type=Path, default=base / "aggregate_data_pv.csv")
    ap.add_argument("--factor", type=float, default=4.0,
                    help="¼→full == 4 (the old GT_DOWNSCALE)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--validate-neurons", type=int, default=5)
    args = ap.parse_args()

    src = Registration.from_json(args.src)
    print(f"[scale] src mean A = {np.round(src.A, 5).tolist()}  (x{args.factor})")
    scaled = scale_registration(src, args.factor)
    print(f"[scale] new mean A = {np.round(scaled.A, 5).tolist()}  "
          f"(expect ~I; affines {None if scaled.affines is None else scaled.affines.shape})")
    scaled.to_json(args.out)
    print(f"[scale] saved -> {args.out}")

    if not args.no_validate:
        gt = GroundTruth.from_config()
        print(f"[scale] validating on {args.validate_neurons} neurons against full_scale GT "
              f"(downscale={gt.downscale}) — reads full-res slices, slow...")
        rate, n = on_mask_rate(gt, args.skeleton_csv, scaled,
                               neuron_limit=args.validate_neurons)
        print(f"[scale] on-mask: {rate:.1%} (n={n})  "
              f"[quarter fit reported 85.7pct over 40 neurons]")


if __name__ == "__main__":
    main()
