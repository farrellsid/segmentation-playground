"""Open several corrected neurons together in napari, each toggleable on/off.

One napari Labels layer per neuron, each positioned by its own translate/scale
so they land correctly in the shared real-world (nanometre) coordinate space,
reusing render_review.neuron_volume for the per-neuron remap/crop (extended, not
rewritten, to also return each neuron's bounding-box origin, see
render_review.py's own docstring).
Overlapping neurons are shown blended by napari's own layer compositing; this
tool does not resolve overlaps into one canonical label (see
docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review


def build_layer_specs(root, neurons: list[str]) -> list[dict]:
    """One layer spec per neuron that actually has masks, in selection order.

    A neuron with an empty volume (no masks found under `root`) is skipped
    entirely, not added as an empty layer. Each spec's `data` is `vol` (from
    `render_review.neuron_volume`) multiplied by that neuron's 1-based selection
    index, so napari's own per-label-value colormap gives every neuron a
    distinct, consistent color with no custom palette. The multiply uses uint16,
    wide enough that a neuron count in the hundreds cannot wrap around.

    `translate` is each neuron's voxel-space `origin` converted to real-world
    nanometres via `spacing` (origin * spacing, elementwise), not the raw voxel
    offset, because `scale=spacing` is also set on the same layer and napari
    composes scale and translate in world units.
    """
    specs = []
    for i, neuron in enumerate(neurons, start=1):
        vol, origin, spacing = render_review.neuron_volume(root, neuron)
        if not vol.size or not vol.any():
            print(f"[view] {neuron}: no masks, skipping")
            continue
        id_vol = vol.astype(np.uint16) * np.uint16(i)
        translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))
        specs.append({"name": neuron, "data": id_vol, "translate": translate,
                      "scale": tuple(float(s) for s in spacing)})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D mask viewer (napari)")
    ap.add_argument("--tree", required=True,
                    help="bundle folder or output tree root (same layout either way)")
    ap.add_argument("--neurons", required=True,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL")
    args = ap.parse_args()

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    specs = build_layer_specs(args.tree, neurons)
    if not specs:
        raise SystemExit(
            f"[view] none of {neurons} have any masks under {args.tree}")

    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    print(f"[view] opened {len(specs)} neuron(s): {', '.join(s['name'] for s in specs)}")
    napari.run()


if __name__ == "__main__":
    main()
