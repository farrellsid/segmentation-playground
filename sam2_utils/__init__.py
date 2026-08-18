"""sam2_utils: shared helpers for SAM2 experiments on Zhen Lab EM data.

Every submodule is imported on demand, not by this file. ``sam2_utils/__init__.py``
imports nothing itself, so ``from sam2_utils import X`` only pulls in whatever X
itself needs, not the union of every submodule's dependencies. That is what lets
a light consumer, such as an exporter that does ``from sam2_utils import registry``,
stay free of torch/cv2/pandas/numpy/matplotlib even though other submodules in
this package need those heavy deps.

    config        - paths, checkpoint registry, affine constants, data/output paths
    setup         - device setup, checkpoint download, predictor build
    viz           - show_mask / show_points / show_box / show_masks / pick_point
    diagnostics   - VRAM/RAM/disk snapshot, cleanup_vram, peak-VRAM probes
    catmaid       - CATMAID API client + annotation fetch
    alignment     - THE coordinate-transform home: affine, tif<->sam, z maps,
                    nm->stack-px, CropWindow, affine fit + grid sampling
    qc            - post-hoc QC metrics + flag rule over a saved mask stack
    review        - read-only proofreading viewer for a finished chain on disk
    video_viz     - overlay/animate/grid an in-RAM video_segments dict
    labels        - per-frame label store (the "label engine"); pure pandas
    review_queue  - work queue + GUI-owned review-status ledger; pure pandas
                    (the napari GUI itself is top-level ``gui.py``, not in this
                    package, to keep napari out of the import path entirely)
    membrane      - membrane-ness map (Sato ridge filter) + boundary detectors
    registry      - permanent cell_name to neuron_id mapping; csv + pathlib only,
                    no torch/cv2/pandas/network, so a bare exporter can import it
"""
