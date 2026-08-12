"""Spot-check: does MitoNet find plausible mitochondria on target-worm EM?

Same role as experiments/nucleonet_spotcheck.py, a characterization spike, not production code.
Lazy-imports torch/empanada so nothing outside this script gains a new heavy dependency. Visual
gut-check only: render MitoNet's detections over the raw EM and read them by eye before trusting
the output for anything downstream (same discipline the NucleoNet spot-check used).

    py -3 experiments/mitonet_spotcheck.py                # z=1456, the current-work frame
    py -3 experiments/mitonet_spotcheck.py --z 1472

Prerequisite: `pip install empanada-dl --no-deps` (see nucleonet_spotcheck.py's docstring for why
`--no-deps` is needed on this environment). The first run downloads the MitoNet checkpoint to
`~/.empanada`, outside this repo, and reuses the cached file on later runs.

Config source: empanada_napari/configs/MitoNet_v1.yaml (from github.com/volume-em/empanada-napari,
fetched 2026-08-04). Unlike NucleoNet's config, this one's `nms_threshold`/`nms_kernel`/
`confidence_thr` provenance IS confirmed (present in the yaml's FINETUNE.engine_params, matching
values used here). `coarse_boundaries` still is not in the yaml; kept at NucleoNet's value pending
a real check against empanada's own inference default.
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

OUT_DIR = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\mitonet-spotcheck")
SCALE = 8

# Vendored the same way as nucleonet_spotcheck.py: constants from empanada_napari/configs/
# MitoNet_v1.yaml, loader/preprocess logic from empanada_napari/utils.py. BSD 3-Clause License,
# Copyright (c) 2021, volume-em. Redistributed here per that license's terms.
MITONET_MODEL_URL = "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1"
MITONET_MEAN = 0.57571
MITONET_STD = 0.12765
MITONET_PADDING_FACTOR = 16
MITONET_LABEL_DIVISOR = 1000
MITONET_THING_LIST = [1]  # class 1 = "mito", the only class this model predicts


def _load_mitonet_weights(device):
    """Download (and cache) the MitoNet TorchScript checkpoint to the same ~/.empanada cache dir
    convention as nucleonet_spotcheck.py's _load_nucleonet_weights."""
    import os
    import urllib.parse
    import torch

    model_dir = os.path.join(os.path.expanduser("~"), ".empanada")
    torch.hub.set_dir(model_dir)
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)

    filename = os.path.basename(urllib.parse.urlparse(MITONET_MODEL_URL).path)
    cached_file = os.path.join(hub_dir, filename)
    if not os.path.exists(cached_file):
        print(f"[mitonet] downloading weights to {cached_file} ...")
        torch.hub.download_url_to_file(MITONET_MODEL_URL, cached_file, None, progress=True)
    else:
        print(f"[mitonet] using cached weights at {cached_file}")

    model = torch.jit.load(cached_file, map_location=device)
    return model.to(device).eval()


def _preprocess(image: np.ndarray):
    """Same normalization as nucleonet_spotcheck.py's _preprocess (MitoNet shares the same
    mean/std in its yaml)."""
    import torch

    max_value = 255.0
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * max_value
    mean = np.float32(MITONET_MEAN) * max_value
    std = np.float32(MITONET_STD) * max_value
    img = (img - mean) * np.reciprocal(std, dtype=np.float32)
    return torch.from_numpy(img[None])


def run_mitonet(em_gray: np.ndarray) -> np.ndarray:
    """Run MitoNet on a single grayscale EM image, return an instance label map,
    same H x W as the input, 0 = background, 1..N = detected mitochondria instances."""
    import torch
    from empanada.inference.engines import PanopticDeepLabRenderEngine

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_mitonet_weights(device)
    engine = PanopticDeepLabRenderEngine(
        model, thing_list=MITONET_THING_LIST,
        label_divisor=MITONET_LABEL_DIVISOR,
        nms_threshold=0.1, nms_kernel=7, confidence_thr=0.5,
        padding_factor=MITONET_PADDING_FACTOR, coarse_boundaries=True,
    )

    size = em_gray.shape
    tensor = _preprocess(em_gray).unsqueeze(0).to(device)  # (1, 1, H, W)
    with torch.no_grad():
        pan_seg = engine(tensor, size, upsampling=1)
    pan_seg = pan_seg.squeeze().cpu().numpy()

    class_map = pan_seg // MITONET_LABEL_DIVISOR
    inst_map = pan_seg % MITONET_LABEL_DIVISOR
    mito_labels = np.where(class_map == MITONET_THING_LIST[0], inst_map, 0).astype(np.int32)
    return mito_labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    em, _full_hw = pipeline.load_frame_sam(args.z, scale=SCALE)
    em_gray = em.mean(axis=2) if em.ndim == 3 else em

    print(f"[mitonet] z={args.z}: running MitoNet inference ...")
    mito_labels = run_mitonet(em_gray.astype(np.float32))
    n_detected = int(len(np.unique(mito_labels)) - 1)
    print(f"[mitonet] z={args.z}: {n_detected} mitochondria instances detected")

    lut = do.build_palette(max(n_detected, 1))
    mito_rgb = do.colorize_over_em(em, mito_labels, lut, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.8))
    em_rgb = np.stack([em_gray.astype(np.uint8)] * 3, axis=2) if em.ndim != 3 else em
    ax[0].imshow(em_rgb); ax[0].set_title(f"raw EM (z={args.z})", fontsize=11)
    ax[1].imshow(mito_rgb); ax[1].set_title(f"MitoNet detections ({n_detected} instances)", fontsize=11)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MitoNet spot-check on target-worm EM", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out_dir / f"mitonet_spotcheck_z{args.z}.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[mitonet] wrote {f}")


if __name__ == "__main__":
    main()
