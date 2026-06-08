"""SAM2 setup: device selection, checkpoint download, predictor construction.

Typical usage in a notebook:

    from sam2_utils import setup, config
    device = setup.setup_device()                            # picks CUDA/MPS/CPU + autocast
    ckpt, cfg = setup.ensure_checkpoint("small")             # downloads if missing
    predictor = setup.build_image_predictor(ckpt, cfg, device)
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Tuple, Literal

from . import config


# =============================================================================
# Device
# =============================================================================

# Module-level handle on the autocast context so it stays alive for the whole
# session (Meta's reference notebooks also rely on this pattern).
_autocast_ctx = None


def setup_device(verbose: bool = True):
    """Select CUDA / MPS / CPU, enter bfloat16 autocast on CUDA, enable TF32 on Ampere+.

    Returns
    -------
    torch.device
        The selected device.
    """
    import torch  # imported lazily so importing this module doesn't require torch

    # MPS fallback for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Reduce CUDA fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Using device: {device}")

    global _autocast_ctx
    if device.type == "cuda":
        if _autocast_ctx is None:
            _autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            _autocast_ctx.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps" and verbose:
        print(
            "Note: MPS support in SAM 2 is preliminary; outputs may differ from CUDA. "
            "See https://github.com/pytorch/pytorch/issues/84936"
        )

    return device


# =============================================================================
# Checkpoint download
# =============================================================================

def _make_progress_hook():
    """Return a urlretrieve reporthook backed by progressbar2, if available."""
    try:
        import progressbar
    except ImportError:
        # Plain fallback: print percent every ~5%
        last = [0]
        def hook(block_num, block_size, total_size):
            if total_size <= 0:
                return
            pct = int(100 * block_num * block_size / total_size)
            if pct >= last[0] + 5:
                last[0] = pct
                print(f"  {min(pct, 100)}%", end="", flush=True)
                if pct >= 100:
                    print()
        return hook

    state = {"bar": None}
    def hook(block_num, block_size, total_size):
        if state["bar"] is None:
            state["bar"] = progressbar.ProgressBar(maxval=total_size)
            state["bar"].start()
        downloaded = block_num * block_size
        if downloaded < total_size:
            state["bar"].update(downloaded)
        else:
            state["bar"].finish()
            state["bar"] = None
    return hook


def ensure_checkpoint(
    size: str = config.DEFAULT_MODEL_SIZE,
    checkpoint_dir: Path | None = None,
) -> Tuple[Path, str]:
    """Download the requested SAM2 checkpoint if not already on disk.

    Parameters
    ----------
    size : {"tiny", "small", "base_plus", "large"}
        Which checkpoint to fetch.
    checkpoint_dir : Path, optional
        Override the default checkpoint directory (config.CHECKPOINT_DIR).

    Returns
    -------
    (checkpoint_path, model_config_path)
        Pass both into `build_image_predictor` / `build_video_predictor`.
    """
    if size not in config.SAM2_CHECKPOINTS:
        raise ValueError(
            f"Unknown SAM2 size {size!r}. "
            f"Available: {sorted(config.SAM2_CHECKPOINTS)}"
        )
    url, filename, model_cfg = config.SAM2_CHECKPOINTS[size]
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else config.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / filename

    if dest.exists():
        print(f"Checkpoint already present: {dest}")
    else:
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest, _make_progress_hook())
        print(f"Saved to {dest}")

    return dest, model_cfg


# =============================================================================
# Predictor build
# =============================================================================

def build_image_predictor(
    checkpoint_path: Path | str,
    model_cfg: str,
    device,
):
    """Build a SAM2ImagePredictor. Build once, reuse for the session."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    return SAM2ImagePredictor(sam2_model)


def build_video_predictor(
    checkpoint_path: Path | str,
    model_cfg: str,
    device,
    correct_as_cond: bool = False,
):
    """Build a SAM2 video predictor for spatio-temporal masklets.

    ``correct_as_cond`` sets SAM2's ``add_all_frames_to_correct_as_cond``: when True,
    a frame that receives a *correction* (a mask/point added after it was already
    tracked) is promoted to a **conditioning** frame, so its corrected mask is stored
    verbatim and re-emitted on the next ``propagate_in_video`` instead of being
    re-inferred from memory (which silently discards the correction). Required by the
    interactive review GUI, where a human-painted mask must be authoritative across
    iterative paint→resume→repaint cycles (PIPELINE_CONTEXT §7 *box vs mask*: "the
    human-painted mask is the maximally-verified seed").

    **Default False** preserves the exact headless build: the batch pipeline only ever
    seeds the anchor (an *initial* conditioning frame, unaffected by this flag) and
    never corrects an already-tracked frame, so the flag is inert there and the M1
    AVAL pixel-for-pixel reproduction is unchanged.
    """
    from sam2.build_sam import build_sam2_video_predictor
    overrides = (["++model.add_all_frames_to_correct_as_cond=true"]
                 if correct_as_cond else [])
    return build_sam2_video_predictor(model_cfg, str(checkpoint_path), device=device,
                                      hydra_overrides_extra=overrides)


def build_predictor(
    size: str = config.DEFAULT_MODEL_SIZE,
    kind: Literal["image", "video"] = "image",
    device=None,
    checkpoint_dir: Path | None = None,
    correct_as_cond: bool = False,
):
    """One-shot convenience: pick size + kind, get a ready-to-use predictor.

    ``correct_as_cond`` (video only) promotes human corrections to conditioning
    frames — see ``build_video_predictor``. Ignored for ``kind="image"``.

    Returns
    -------
    (predictor, device)
        `device` is included so callers don't need to also call setup_device.
    """
    if device is None:
        device = setup_device(verbose=True)
    ckpt, cfg = ensure_checkpoint(size, checkpoint_dir=checkpoint_dir)
    if kind == "image":
        predictor = build_image_predictor(ckpt, cfg, device)
    elif kind == "video":
        predictor = build_video_predictor(ckpt, cfg, device, correct_as_cond=correct_as_cond)
    else:
        raise ValueError(f"kind must be 'image' or 'video', got {kind!r}")
    return predictor, device


# =============================================================================
# SAM2 availability check
# =============================================================================

def check_sam2_available(auto_install: bool = False) -> bool:
    """Test whether the sam2 package can be imported.

    If auto_install is True and sam2 is missing, pip-installs it from GitHub.
    """
    try:
        import sam2  # noqa: F401
        print("SAM 2 found.")
        return True
    except ImportError:
        print("SAM 2 not installed.")
        if auto_install:
            print("Installing from GitHub...")
            os.system(
                f"{sys.executable} -m pip install "
                "'git+https://github.com/facebookresearch/sam2.git'"
            )
            try:
                import sam2  # noqa: F401
                return True
            except ImportError:
                pass
        return False
