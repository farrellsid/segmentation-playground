"""sam2_utils — shared helpers for SAM2 experiments on Zhen Lab EM data.

Eagerly imported (available as ``sam2_utils.<name>`` after ``import sam2_utils``):
    config        - paths, checkpoint registry, affine constants, data/output paths
    setup         - device setup, checkpoint download, predictor build
    viz           - show_mask / show_points / show_box / show_masks / pick_point
    diagnostics   - VRAM/RAM/disk snapshot, cleanup_vram, peak-VRAM probes
    catmaid       - CATMAID API client + annotation fetch
    alignment     - THE coordinate-transform home: affine, tif<->sam, z maps,
                    nm->stack-px, CropWindow, affine fit + grid sampling

Import-on-demand (heavier deps - skimage/scipy/matplotlib/napari; do
``from sam2_utils import qc`` etc. so they're not pulled into light imports, and
so pipeline.py can stay torch/qc-free at import time):
    qc            - post-hoc QC metrics + flag rule over a saved mask stack
    review        - read-only proofreading viewer for a finished chain on disk
    video_viz     - overlay/animate/grid an in-RAM video_segments dict
    labels        - M4 per-frame label store (the "label engine"); pure pandas
    review_queue  - M4 work queue + GUI-owned review-status ledger; pure pandas
                    (the napari GUI itself is top-level ``gui.py``, not in this
                    package, to keep napari out of the import path entirely)
"""

# Only the light/core modules are imported eagerly. The heavy viewers (qc,
# review, video_viz) are intentionally left out so `import sam2_utils` and
# pipeline.py's `from sam2_utils import config, alignment` don't drag in
# skimage/scipy/matplotlib; import them explicitly when needed.
from . import config, setup, viz, diagnostics, catmaid, alignment

__all__ = ["config", "setup", "viz", "diagnostics", "catmaid", "alignment"]
