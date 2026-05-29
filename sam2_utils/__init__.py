"""sam2_utils — shared helpers for SAM2 experiments on Zhen Lab EM data.

Submodules:
    config        — paths, checkpoint registry, affine constants
    setup         — device setup, checkpoint download, predictor build
    viz           — show_mask / show_points / show_box / show_masks / pick_point
    diagnostics   — VRAM/RAM/disk snapshot, cleanup_vram
    catmaid       — CATMAID API client + annotation fetch
    alignment     — affine fit/apply, grid landmark sampling
"""

from . import config, setup, viz, diagnostics, catmaid, alignment

__all__ = ["config", "setup", "viz", "diagnostics", "catmaid", "alignment"]
