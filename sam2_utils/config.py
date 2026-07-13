"""Configuration constants for SAM2 experiments.

Keep this module import-light: no torch, no cv2, no network calls.
That way `from sam2_utils import config` is instant and safe in any context.
"""

import os
from pathlib import Path

# =============================================================================
# Filesystem paths
# =============================================================================

#: Root of the raw EM data (sorted .tif stack). Override with the SAM2_WORM_PATH env
#: var (used on the cluster, where the stack lives under project storage); unset, it
#: falls back to the local Windows default so existing local runs are unaffected.
#: Local default is F: (the original E: drive's cable failed, same migration as the
#: GT paths below; F: contents match the old E: layout).
WORM_PATH = Path(os.environ.get("SAM2_WORM_PATH", r"F:\ZhenLab\Data\SAM2_test_NR_raw"))

#: Where SAM2 checkpoints are downloaded to. Relative to notebook CWD by default.
CHECKPOINT_DIR = Path("checkpoints")

# -----------------------------------------------------------------------------
# Pipeline data + output paths (one home, imported by run_aval.py / batch.py)
# -----------------------------------------------------------------------------
# DATA_DIR is derived from this file's location, so the in-repo CSV/JSON resolve
# wherever the repo is checked out (no hardcoded D:\ path). OUTPUT_ROOT and
# FRAMES_ROOT are machine-specific scratch/output on a fast volume, like
# WORM_PATH above; edit them for your box.

#: The repo's data/ dir (aggregate_data_pv.csv, chains.json, roots.json).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH    = DATA_DIR / "aggregate_data_pv.csv"   #: cached CATMAID node table
CHAINS_PATH = DATA_DIR / "chains.json"             #: per-neuron MLC chains
ROOTS_PATH  = DATA_DIR / "roots.json"              #: chain roots

#: Where per-chain mask outputs + the manifest/triage CSVs are written.
OUTPUT_ROOT = Path(r"F:\ZhenLab\Data\output_masks\test2_single")
#: Parent dir for the SAM2 JPEG frame cache + per-chain link views.
FRAMES_ROOT = Path(r"F:\ZhenLab\Data")

# -----------------------------------------------------------------------------
# Cross-worm ground truth (eval/)  (see eval/README.md)
# -----------------------------------------------------------------------------
# The step-back GT: a *different* worm (SEM-Dauer 1), manually segmented
# in VAST, with matching EM. (The target/production worm, CATMAID project 336,
# WORM_PATH above, is the "sensory ablated dauer"; SEM-Dauer 1 is project 280.)
# The VAST export is a 16-bit single-channel LABELMAP per z-slice: pixel value ==
# segment number (`Nr`) in GT_METADATA, 0 == Background (see eval/groundtruth.py).
# Masks + EM are downscaled GT_DOWNSCALE× from the full-res VAST stack the metadata
# bboxes/anchors are quoted in. Machine-specific (like WORM_PATH / OUTPUT_ROOT
# above); edit for your box.
#
# On the 2TB F: HDD (the original E: drive's cable failed). F: is stable, so
# the old local-copy-to-repo strategy (GT_*_LOCAL below) is no longer needed:
# point straight at F:. The F: contents are byte-identical to the old E: drop.
#
# The FULL-RESOLUTION GT is exported (full_scale/ next to
# one_fourth_scale/: 851 slices, 9728×9216, == the metadata's full-res VAST coord
# grid), so GT_DOWNSCALE is 1 and the paths point at full_scale. The full-res
# registration.json is in place (A ≈ I): produced by scaling the ¼-scale fit ×4 via
# `py -3 -m eval.scale_registration` (geometrically exact, instant vs a ~1.5 h
# from-scratch `eval.registration` re-fit; ¼ fit kept as registration_quarter_scale.json).
# The one_fourth_scale dirs remain on F: as a fallback.

#: Root of the cross-worm GT drop (F: 2TB HDD).
GT_ROOT = Path(r"F:\Zhen Lab\SEM DAUER 1")
#: VAST per-slice segmentation labelmaps (*_s###.png, mode I;16, full res 9728×9216).
GT_MASK_DIR = GT_ROOT / "Segmentations" / "full_scale"
#: Matching full-res EM slices (*_s###.png).
GT_EM_DIR = GT_ROOT / "RAW IMAGES" / "full_scale"
#: VAST-Lite extended color/segment file: the Nr↔name↔bbox table.
GT_METADATA = GT_ROOT / "VAST_segmentation_metadata.txt"
#: Downscale of the exported masks/EM vs the full-res VAST coords used in the
#: metadata bboxes/anchors. full_scale == full res → 1 (one_fourth_scale was 4).
GT_DOWNSCALE = 1
# (The flaky-E:-era local-copy constants GT_MASK_DIR_LOCAL / GT_METADATA_LOCAL were
#  removed when the GT moved to the stable F: drive; from_config reads F: directly.)

# -- Prediction on GT worm (pred pipeline, eval/predict_gt.py) -----------------
# Running SAM2 on SEM-Dauer 1's EM (seeded from the p280 skeletons via the fitted
# registration) writes predicted masks onto the GT grid here. Two views are
# emitted from one run (see eval/predict_gt.py): per-neuron binary masks for
# eval.score.DirPredictionSource, and per-slice labelmaps for eval.run_erl --pred.
#: Root for predicted output on SEM-Dauer 1. ``masks/<neuron>/<slice:03d>.png``
#: (binary, for score.py) and ``labelmaps/*_s###.png`` (uint16, for run_erl pred).
GT_PRED_DIR = DATA_DIR / "groundtruth" / "pred_p280"


# =============================================================================
# SAM2 model registry
# =============================================================================
# Maps a size key to (download_url, local_filename, hydra config path).
# Config paths are resolved by sam2 internally relative to its package, so the
# `configs/sam2.1/...` prefix is correct as-is; don't make it absolute.

SAM2_CHECKPOINTS = {
    "tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
    "small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "base_plus": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "large": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_large.pt",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
}

#: Default model size if not specified.
DEFAULT_MODEL_SIZE = "large"


# =============================================================================
# CATMAID
# =============================================================================

CATMAID_URL = "https://zhencatmaid.com/"
CATMAID_PROJECT_ID = 336

def get_catmaid_token() -> str:
    """Read the CATMAID API token from the CATMAID_TOKEN env var.

    Falls back to reading a .env file next to this config if the env var
    isn't set. Raises if neither is available.
    """
    tok = os.environ.get("CATMAID_TOKEN")
    if tok:
        return tok
    # tiny .env fallback (KEY=VALUE per line, comments with #)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "CATMAID_TOKEN":
                    return v.strip().strip('"').strip("'")
    raise RuntimeError(
        "CATMAID_TOKEN not found. Set it in your environment "
        "(e.g. `export CATMAID_TOKEN=...`) or put it in a .env file "
        "next to the sam2_utils package."
    )


# =============================================================================
# CATMAID stack -> tif alignment
# =============================================================================
# Affine fit from the 12-landmark set at CATMAID z=1293.
# See CATMAID_alignment.ipynb for the derivation.

import numpy as _np  # local-only, doesn't pollute module namespace

#: Affine matrix mapping CATMAID stack-px coords to tif-px coords.
M_AFFINE = _np.array([
    [ 1.11667,  0.03757],
    [-0.03641,  1.11909],
])

#: Translation component of the same affine.
T_AFFINE = _np.array([-893.90, -427.09])

#: Filename z vs CATMAID z offset. File "z1300" corresponds to CATMAID z=1293.
#: I.e. CATMAID_z = file_z + FILE_Z_OFFSET.
FILE_Z_OFFSET = -7

#: Stack resolution in nm/voxel (x, y, z). Used to convert CATMAID API
#: nm coordinates to stack-pixel coordinates.
STACK_RESOLUTION_NM = (2, 2, 50)
