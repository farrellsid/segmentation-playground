"""Configuration constants for SAM2 experiments.

Keep this module import-light: no torch, no cv2, no network calls.
That way `from sam2_utils import config` is instant and safe in any context.
"""

import os
from pathlib import Path

# =============================================================================
# Filesystem paths
# =============================================================================

#: Root of the raw EM data (sorted .tif stack).
WORM_PATH = Path(r"E:\ZhenLab\Data\SAM2_test_NR_raw") # Change this to wherever path may be.

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
OUTPUT_ROOT = Path(r"E:\ZhenLab\Data\output_masks\test2_single")
#: Parent dir for the SAM2 JPEG frame cache + per-chain link views.
FRAMES_ROOT = Path(r"E:\ZhenLab\Data")


# =============================================================================
# SAM2 model registry
# =============================================================================
# Maps a size key to (download_url, local_filename, hydra config path).
# Config paths are resolved by sam2 internally relative to its package, so the
# `configs/sam2.1/...` prefix is correct as-is — don't make it absolute.

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
DEFAULT_MODEL_SIZE = "small"


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
