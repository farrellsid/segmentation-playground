"""Frame source seam: logical z -> EM file + cache, and the frame-read helpers.

run_chain's phase functions are otherwise worm-agnostic (they consume annotate_df's
x_tif/y_tif + a logical z), but the EM frames are read straight from the target worm's
tif stack by file_z. A FrameStore abstracts "logical z -> source EM file + an integer
cache/order key", so the same pipeline can run on a different worm (e.g. SEM-Dauer 1's
per-slice PNG export) by passing a different store to run_chain. The default
(TifFrameStore) reproduces the original target-worm behavior byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sam2_utils import config, alignment


# =============================================================================
# Module-local helpers (shared across phases -> defined once, used twice)
# =============================================================================

def _parse_file_z(p) -> int:
    """'.../1301____z1300.0.tif' -> 1300. Matches the notebook's parse_file_z."""
    token = Path(p).stem.split("z")[-1]
    return int(float(token))


def _downscale_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Area-downsample by `scale`. scale==1 is a no-op copy (notebook helper)."""
    import cv2
    if scale == 1:
        return img.copy()
    return cv2.resize(img, None, fx=1 / scale, fy=1 / scale,
                      interpolation=cv2.INTER_AREA)


def _read_tif_window(tif_path, sl) -> np.ndarray:
    """Return the [y0:y1, x0:x1] window `sl` of a tif as BGR HxWx3 uint8, EXACTLY what
    ``cv2.imread(str(tif_path))[sl]`` returns, but read lazily so only the window's rows
    page in instead of decoding the whole frame (the tier-2 perf optimisation).

    The Zhen EM tifs are uncompressed, single-strip (row-contiguous), 8-bit grayscale, so a
    ``tifffile.memmap`` slice touches only the sliced rows: ~(y1-y0)*W bytes vs the full
    ~85 MB frame. The window is COPIED out (np.array) so the file mapping is released and the
    returned array is plain in-memory. cv2.imread loads grayscale as 3-channel BGR by
    replication, so GRAY2BGR reproduces it bit-for-bit (and BGR==RGB here anyway since the
    source is grayscale). Any tif that can't be windowed this way (compressed, tiled,
    multi-page/3-channel, or tifffile missing) falls through to a full cv2.imread+slice, so
    the output is invariant to the read path, only the wall-time differs."""
    import cv2
    try:
        import tifffile
        mm = tifffile.memmap(str(tif_path), mode="r")
        try:
            if mm.ndim != 2:                         # only 2D grayscale is windowable here
                raise ValueError("not a 2D grayscale tif")
            win = np.array(mm[sl])                    # copy of just the window (pages in its rows)
        finally:
            del mm                                    # release the mapping
        return cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)  # -> BGR 3-ch, matching cv2.imread
    except Exception:
        return cv2.imread(str(tif_path))[sl]          # safe full-read fallback


def _ensure_cached_frames(subset, cache_dir: Path, scale: int) -> None:
    """Decode+downscale any source frames not yet in the shared cache.

    `subset` is a list of ``(key, src_path)`` from a FrameStore, one JPEG per `key`,
    written once ever at this `scale`, named ``z{key}.jpg`` and reused by every chain
    whose z-range overlaps it. This is where the prep cost actually lives (a ~9k x 9k
    imread + resize); overlapping chains now pay it once across the whole dataset
    instead of once per chain. (`key` == file_z for the tif store, == slice z for the
    GT png store; the cache name scheme is unchanged for the target worm.)
    """
    import cv2
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [(k, p) for (k, p) in subset
               if not (cache_dir / f"z{k}.jpg").exists()]
    if not missing:
        return
    for key, src_path in tqdm(missing, desc="caching JPEG frames", unit="frame"):
        img = cv2.imread(str(src_path))          # BGR, fine for grayscale EM (tif or png)
        img = _downscale_image(img, scale)       # match image-mode coord space
        cv2.imwrite(str(cache_dir / f"z{key}.jpg"), img)


def _link_frame(src: Path, dst: Path) -> None:
    """Expose cache frame `src` at 0-indexed view path `dst`.

    Tries symlink, then hard-link, then a plain copy. On Windows bare symlinks
    need Developer Mode or admin, so the hard-link branch is the usual one -> it
    requires src and dst on the same volume (both live under frames_root, so OK).
    """
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            import os
            os.link(src, dst)                    # hard-link fallback (no privilege)
        except OSError:
            import shutil
            shutil.copy2(src, dst)               # last resort


# =============================================================================
# Frame source, the ONE worm-coupled seam
# =============================================================================

class FrameStore:
    """Maps a *logical z* (catmaid_z on the target worm; VAST slice z on SEM-Dauer 1)
    to its source EM file, and assigns each frame a stable integer ``key`` used for the
    shared JPEG cache name and the frame ordering. Subclass to retarget the pipeline's
    EM source without touching run_chain or any phase function."""

    def key_of_z(self, z: int) -> int:
        raise NotImplementedError

    def z_of_key(self, key: int) -> int:
        raise NotImplementedError

    def file_for_z(self, z: int) -> Path:
        """The single source file for logical z (the anchor-frame load)."""
        raise NotImplementedError

    def files_in_z_range(self, z0: int, z1: int) -> "list[tuple[int, Path]]":
        """``[(key, path), ...]`` for every frame with logical z in [z0, z1], sorted by key."""
        raise NotImplementedError


class TifFrameStore(FrameStore):
    """Target worm: a ``.tif`` stack under ``worm_path`` named ``..z{file_z}.tif``.
    key == file_z, logical z == catmaid_z (related by config.FILE_Z_OFFSET via alignment).
    Reproduces the original glob/parse/z-map exactly, the reproduction path."""

    def __init__(self, worm_path: Optional[Path] = None):
        self.worm_path = Path(worm_path) if worm_path is not None else config.WORM_PATH

    def key_of_z(self, z: int) -> int:
        return alignment.catmaid_z_to_file_z(int(z))

    def z_of_key(self, key: int) -> int:
        return alignment.file_z_to_catmaid_z(int(key))

    def file_for_z(self, z: int) -> Path:
        k = self.key_of_z(z)
        matches = [f for f in self.worm_path.glob("*.tif") if _parse_file_z(f) == k]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected 1 tif for file_z={k} (z={z}), got {len(matches)}: {matches}")
        return matches[0]

    def files_in_z_range(self, z0: int, z1: int) -> "list[tuple[int, Path]]":
        k0, k1 = self.key_of_z(z0), self.key_of_z(z1)
        lo, hi = (k0, k1) if k0 <= k1 else (k1, k0)
        out = [(_parse_file_z(f), f) for f in self.worm_path.glob("*.tif")]
        return sorted([(k, f) for (k, f) in out if lo <= k <= hi], key=lambda kf: kf[0])


def load_frame_sam(catmaid_z: int, *, scale: int,
                   frame_store: Optional[FrameStore] = None
                   ) -> tuple[np.ndarray, tuple[int, int]]:
    """Find the EM frame for logical `catmaid_z`, read it, downscale by `scale`.

    Returns (image_sam RGB uint8, full_hw) -> full_hw is the pre-downscale (H, W),
    kept only so later steps can map back to full-res if ever needed.

    `frame_store` selects the EM source; the default (TifFrameStore) is the original
    tif-stack path. Lift from: parse_file_z + tif glob + cv2.imread + downscale_image.
    """
    import cv2

    fs = frame_store or TifFrameStore()
    src_path = fs.file_for_z(catmaid_z)

    image_full = cv2.cvtColor(cv2.imread(str(src_path)), cv2.COLOR_BGR2RGB)
    H_full, W_full = image_full.shape[:2]
    image_sam = _downscale_image(image_full, scale)
    return image_sam, (H_full, W_full)
