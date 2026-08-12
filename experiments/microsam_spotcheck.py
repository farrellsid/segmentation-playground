"""Spot-check: does micro_sam's EM-boundary checkpoint find a plausible neurite mask
on target-worm EM, given the SAME skeleton-node point prompt our own SAM2 pipeline uses?

A characterization spike, not production code (matches nucleonet_spotcheck.py /
mitonet_spotcheck.py's role). No finetuning on our side: `vit_b_em_boundaries` is
micro_sam's own EM-boundary-trained checkpoint (their doc description: "compartments
delineated by boundaries such as cells or neurites in EM"), used zero-shot. Renders
our SAM2 mask and micro_sam's mask side by side on the same crop, same single positive
point, no negatives on either side (a fair single-point comparison); the caller reads
the pair by eye, same as the NucleoNet/MitoNet spot-checks before a formal metric.

    py -3 experiments/microsam_spotcheck.py --z 1456 --neuron AIYL
    py -3 experiments/microsam_spotcheck.py --z 1472 --neuron AIYL
    py -3 experiments/microsam_spotcheck.py --z 1456 --neuron AIYL --model-type vit_b_em_organelles

Prerequisite: `pip install micro_sam` (installed clean on this environment's Python
3.13, no dependency pin fights like NucleoNet's numpy==1.22 issue). The first run
downloads the requested checkpoint via micro_sam's own cache (torch.hub under the
hood); later runs reuse it.

Checkpoint note: an earlier pass of this spike assumed micro_sam shipped an
`em_boundaries` checkpoint (its own documentation describes one, "compartments
delineated by boundaries such as cells or neurites in EM"), but the installed
registry (`micro_sam.util.models()`, checked directly, not assumed) only has
`vit_{t,b,l}_em_organelles`, no `em_boundaries` variant. `em_organelles` is the only
EM-domain checkpoint actually available right now, so that is what this script runs
by default; it is mitochondria/organelle-trained, not neurite-trained, which is the
exact case our own roadmap's caveat ("an EM finetune trained on organelles is
detrimental for neurites", from the micro_sam paper itself) already predicts might
go badly here. That prediction is part of what this spot-check is testing, not an
assumption baked into the script.
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
from pipeline.predict import image_predict
from pipeline.state import Prompts
from sam2_utils import setup as sam2_setup
from eval.merge_metric import load_node_table

OUT_DIR = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\microsam-spotcheck")
SCALE = 8
DEFAULT_MODEL_TYPE = "vit_l_em_organelles"


def pick_node(annotate_df, z: int, neuron: str | None):
    """A real skeleton node at CATMAID z, on the _sam grid. First match, or first
    match for `neuron` when given."""
    df = annotate_df[annotate_df["z"] == z]
    if neuron is not None:
        df = df[df["cell_name"] == neuron]
    if df.empty:
        raise SystemExit(f"no node found at z={z}" + (f" for {neuron}" if neuron else ""))
    row = df.iloc[0]
    x_sam = float(row["x_tif"]) / SCALE
    y_sam = float(row["y_tif"]) / SCALE
    return str(row["cell_name"]), x_sam, y_sam


def run_microsam(image_gray: np.ndarray, point_xy: tuple[float, float], model_type: str):
    """Point-prompt a micro_sam checkpoint on one grayscale EM image, single
    positive point, no negatives. Returns a bool mask, same HxW as the input."""
    from micro_sam.util import get_sam_model
    from micro_sam.prompt_based_segmentation import segment_from_points

    # CPU, not the default CUDA autodetect: on this GPU the checkpoint runs in
    # bfloat16, and the installed segment_anything predictor's .cpu().numpy() call
    # has no bfloat16 support (numpy itself doesn't have the dtype), a real
    # compatibility bug between micro_sam's model and this torch/numpy version, not
    # something to patch in the installed package for a one-frame spot-check.
    predictor = get_sam_model(model_type=model_type, device="cpu")
    image_rgb = np.stack([image_gray] * 3, axis=-1)
    predictor.set_image(image_rgb)
    points = np.array([point_xy], dtype=float)
    labels = np.array([1], dtype=int)
    mask = segment_from_points(predictor, points, labels, multimask_output=False)
    return np.asarray(mask).astype(bool).squeeze()


def run_sam2(image_sam: np.ndarray, point_xy: tuple[float, float]):
    """Our own SAM2 image-mode predictor, same single positive point, no
    negatives, so the comparison isolates the backbone, not the prompt."""
    image_predictor, _ = sam2_setup.build_predictor(size="large", kind="image")
    prompts = Prompts(points_sam=np.array([point_xy], dtype=float),
                       labels=np.array([1], dtype=int))
    mask, score, _ = image_predict(image_predictor, image_sam, prompts)
    return mask, score


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--neuron", default=None)
    ap.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    annotate_df = load_node_table()
    neuron, x_sam, y_sam = pick_node(annotate_df, args.z, args.neuron)
    print(f"[microsam] z={args.z} neuron={neuron} node at sam=({x_sam:.1f}, {y_sam:.1f})")

    em, _full_hw = pipeline.load_frame_sam(args.z, scale=SCALE)
    em_gray = (em.mean(axis=2) if em.ndim == 3 else em).astype(np.uint8)
    em_rgb = np.stack([em_gray] * 3, axis=2) if em.ndim != 3 else em

    print("[microsam] running our SAM2 image-mode predictor ...")
    sam2_mask, sam2_score = run_sam2(em, (x_sam, y_sam))
    print(f"[microsam] SAM2: area={int(sam2_mask.sum())} score={sam2_score:.3f}")

    print(f"[microsam] running micro_sam ({args.model_type}) ...")
    microsam_mask = run_microsam(em_gray, (x_sam, y_sam), args.model_type)
    print(f"[microsam] micro_sam: area={int(microsam_mask.sum())}")

    # A whole _sam frame is thousands of px on a side; a ~400px mask is invisible at
    # that scale, so zoom to a crop around the node (padded past the larger of the
    # two masks' extents, so neither gets clipped).
    def _bbox(m):
        ys, xs = np.where(m)
        return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None

    pad = 40
    half = 90
    x0c, x1c = x_sam - half, x_sam + half
    y0c, y1c = y_sam - half, y_sam + half
    for m in (sam2_mask, microsam_mask):
        bb = _bbox(m)
        if bb is None:
            continue
        bx0, bx1, by0, by1 = bb
        x0c, x1c = min(x0c, bx0 - pad), max(x1c, bx1 + pad)
        y0c, y1c = min(y0c, by0 - pad), max(y1c, by1 + pad)
    H, W = em_gray.shape[:2]
    x0c, x1c = max(0, int(x0c)), min(W, int(x1c))
    y0c, y1c = max(0, int(y0c)), min(H, int(y1c))
    crop_slice = (slice(y0c, y1c), slice(x0c, x1c))
    em_crop = em_rgb[crop_slice]
    sam2_crop = sam2_mask[crop_slice]
    microsam_crop = microsam_mask[crop_slice]
    xs_c, ys_c = x_sam - x0c, y_sam - y0c

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].imshow(em_crop)
    ax[0].set_title(f"raw EM (z={args.z})", fontsize=11)
    ax[1].imshow(em_crop)
    ax[1].imshow(np.ma.masked_where(~sam2_crop, sam2_crop), cmap="Blues", alpha=0.55, vmin=0, vmax=1)
    ax[1].set_title(f"our SAM2 (score {sam2_score:.2f}, area {int(sam2_mask.sum())})", fontsize=11)
    ax[2].imshow(em_crop)
    ax[2].imshow(np.ma.masked_where(~microsam_crop, microsam_crop), cmap="Oranges", alpha=0.55, vmin=0, vmax=1)
    ax[2].set_title(f"micro_sam {args.model_type} (area {int(microsam_mask.sum())})", fontsize=11)
    for a in ax:
        a.plot(xs_c, ys_c, marker="*", color="lime", markersize=16, markeredgecolor="black")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"micro_sam vs SAM2 spot-check, {neuron} z={args.z}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = out_dir / f"microsam_spotcheck_{neuron}_z{args.z}_{args.model_type}.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[microsam] wrote {f}")


if __name__ == "__main__":
    main()
