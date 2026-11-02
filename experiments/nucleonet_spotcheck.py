"""Spot-check: does NucleoNet find plausible C. elegans nuclei on target-worm EM?

A characterization spike, not production code (matches experiments/sam3_probe.py's role for the
SAM3 adapters). Lazy-imports torch/empanada so nothing outside this script gains a new heavy
dependency. No ground truth exists yet (that's what the new "nucleus" GUI error type, Task 1, starts
collecting), so this is a visual gut-check: render NucleoNet's detections over the raw EM and read
them by eye, the same way Stage 0.1's registration overlay was judged before a formal metric existed.

    py -3 experiments/nucleonet_spotcheck.py                # z=1456, the current-work frame
    py -3 experiments/nucleonet_spotcheck.py --z 1472

Prerequisite: `pip install empanada-dl --no-deps` (plain `pip install empanada-dl` fails on this
environment's Python 3.13, since it pins numpy==1.22 and no matching wheel exists). The first run
downloads the NucleoNet checkpoint to `~/.empanada`, outside this repo, and reuses the cached file
on later runs.

API discovery notes (see .git/sdd/task-2-report.md for the full trail): the `empanada-dl` PyPI
package (the headless base library, not the `empanada-napari` GUI plugin) ships the model
architectures and the `PanopticDeepLabRenderEngine` inference engine, but NOT a model zoo or a
NucleoNet loader; that registry lives only in the `empanada-napari` GitHub repo as per-model YAML
configs (e.g. `empanada_napari/configs/NucleoNet_base_v2.yaml`), each pointing at a Zenodo-hosted
TorchScript `.pth`. The GUI plugin's `empanada_napari.utils.load_model_to_device` /
`Preprocessor` (which do the download + `torch.jit.load` + normalize) have no napari/Qt import at
module scope, so their logic is small enough to vendor directly below instead of installing
`empanada-napari` (and its napari/Qt dependency tree) just for two helper functions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from experiments import dense_overlay as do

OUT_DIR = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\nucleonet-spotcheck")
SCALE = 8

# The constants and functions below (the NUCLEONET_* values and the two helper functions,
# _load_nucleonet_weights and _preprocess) are adapted from empanada-napari (github.com/volume-em/
# empanada-napari): the constants from empanada_napari/configs/NucleoNet_base_v2.yaml, the
# functions from empanada_napari/utils.py. BSD 3-Clause License, Copyright (c) 2021, volume-em.
# Redistributed here per that license's terms; see docs/superpowers/specs/
# 2026-07-29-nucleus-detector-design.md for the license verification this project did before use.

# NucleoNet_base_v2, from empanada-napari's model registry (config content fetched directly from
# https://raw.githubusercontent.com/volume-em/empanada-napari/main/empanada_napari/configs/
# NucleoNet_base_v2.yaml on 2026-07-29; the base empanada-dl package has no such registry itself).
NUCLEONET_MODEL_URL = (
    "https://zenodo.org/records/18142651/files/"
    "PanopticDeepLabPR_5f023685-7433-404c-9f3d-692898aff9ab_panoptic_deeplab_pointrend.pth"
    "?download=1"
)
NUCLEONET_MEAN = 0.57571
NUCLEONET_STD = 0.12765
NUCLEONET_PADDING_FACTOR = 512
NUCLEONET_LABEL_DIVISOR = 1000
NUCLEONET_THING_LIST = [1]  # class 1 = "nuclei" (the only class this model predicts)


def _load_nucleonet_weights(device):
    """Download (and cache) the NucleoNet TorchScript checkpoint, same cache dir convention
    empanada-napari uses (~/.empanada, via torch.hub), so a manual empanada-napari install would
    reuse the same cached file instead of downloading twice."""
    import os
    import urllib.parse
    import torch

    model_dir = os.path.join(os.path.expanduser("~"), ".empanada")
    torch.hub.set_dir(model_dir)
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)

    filename = os.path.basename(urllib.parse.urlparse(NUCLEONET_MODEL_URL).path)
    cached_file = os.path.join(hub_dir, filename)
    if not os.path.exists(cached_file):
        print(f"[nucleonet] downloading weights to {cached_file} ...")
        torch.hub.download_url_to_file(NUCLEONET_MODEL_URL, cached_file, None, progress=True)
    else:
        print(f"[nucleonet] using cached weights at {cached_file}")

    model = torch.jit.load(cached_file, map_location=device)
    return model.to(device).eval()


def _preprocess(image: np.ndarray):
    """Vendored from empanada_napari.utils.Preprocessor + normalize + to_tensor: scale to
    [0, 255], z-score with NucleoNet's training mean/std, and return a (1, H, W) float tensor."""
    import torch

    max_value = 255.0
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * max_value
    mean = np.float32(NUCLEONET_MEAN) * max_value
    std = np.float32(NUCLEONET_STD) * max_value
    img = (img - mean) * np.reciprocal(std, dtype=np.float32)
    return torch.from_numpy(img[None])


def run_nucleonet(em_gray: np.ndarray) -> np.ndarray:
    """Run NucleoNet on a single grayscale EM image, return an instance label map,
    same H x W as the input, 0 = background, 1..N = detected nucleus instances.

    Lazy-imports torch/empanada so this stays out of the CPU-only test path.

    Confirmed API (see module docstring / task-2-report.md for the discovery trail):
    `empanada.inference.engines.PanopticDeepLabRenderEngine(model, thing_list=..., ...)` wraps a
    TorchScript model and is called as `engine(image_tensor[1,1,H,W], (H, W), upsampling=1)`,
    returning a panoptic label map where each pixel is `class_id * label_divisor + instance_id`
    (0 = void/background). NucleoNet has exactly one class (1 = "nuclei"), so decoding to a plain
    instance map is `pan_seg % label_divisor` wherever `pan_seg // label_divisor == 1`, else 0.
    Verified end-to-end on a random 512x512 array before running on real EM (see report)."""
    import torch
    from empanada.inference.engines import PanopticDeepLabRenderEngine

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_nucleonet_weights(device)
    engine = PanopticDeepLabRenderEngine(
        model, thing_list=NUCLEONET_THING_LIST,
        label_divisor=NUCLEONET_LABEL_DIVISOR,
        # nms_threshold/nms_kernel/confidence_thr/coarse_boundaries: unlike the NUCLEONET_*
        # constants above, this task's discovery pass did not confirm whether these four came
        # from NucleoNet_base_v2.yaml or are empanada_napari.inference.Engine2d's own inference
        # defaults carried over unmodified. Re-check against the yaml or Engine2d's source
        # before treating these as validated hyperparameters beyond this spot-check.
        nms_threshold=0.1, nms_kernel=7, confidence_thr=0.5,
        padding_factor=NUCLEONET_PADDING_FACTOR, coarse_boundaries=True,
    )

    size = em_gray.shape
    tensor = _preprocess(em_gray).unsqueeze(0).to(device)  # (1, 1, H, W)
    with torch.no_grad():
        pan_seg = engine(tensor, size, upsampling=1)
    pan_seg = pan_seg.squeeze().cpu().numpy()

    class_map = pan_seg // NUCLEONET_LABEL_DIVISOR
    inst_map = pan_seg % NUCLEONET_LABEL_DIVISOR
    nuc_labels = np.where(class_map == NUCLEONET_THING_LIST[0], inst_map, 0).astype(np.int32)
    return nuc_labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    em, _full_hw = pipeline.load_frame_sam(args.z, scale=SCALE)
    em_gray = em.mean(axis=2) if em.ndim == 3 else em

    print(f"[nucleonet] z={args.z}: running NucleoNet inference ...")
    nuc_labels = run_nucleonet(em_gray.astype(np.float32))
    # Instance ids come from a modulo decode (pan_seg % label_divisor) and are not guaranteed
    # contiguous, so count distinct ids rather than trusting the max id (subtract 1 for the
    # background/0 label, which is always present).
    n_detected = int(len(np.unique(nuc_labels)) - 1)
    print(f"[nucleonet] z={args.z}: {n_detected} nucleus instances detected")

    lut = do.build_palette(max(n_detected, 1))
    nuc_rgb = do.colorize_over_em(em, nuc_labels, lut, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.8))
    em_rgb = np.stack([em_gray.astype(np.uint8)] * 3, axis=2) if em.ndim != 3 else em
    ax[0].imshow(em_rgb); ax[0].set_title(f"raw EM (z={args.z})", fontsize=11)
    ax[1].imshow(nuc_rgb); ax[1].set_title(f"NucleoNet detections ({n_detected} instances)", fontsize=11)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("NucleoNet spot-check on target-worm EM", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out_dir / f"nucleonet_spotcheck_z{args.z}.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[nucleonet] wrote {f}")


if __name__ == "__main__":
    main()
