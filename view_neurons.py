"""Open several corrected neurons together in napari, each toggleable on/off.

One napari Labels layer per neuron, each positioned by its own translate/scale
so they land correctly in the shared real-world (nanometre) coordinate space,
reusing render_review.neuron_volume for the per-neuron remap/crop (extended, not
rewritten, to also return each neuron's bounding-box origin, see
render_review.py's own docstring).
Overlapping neurons are shown blended by napari's own layer compositing; this
tool does not resolve overlaps into one canonical label (see
docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

With --mesh, each neuron's 3D surface is also built (reusing
sam2_utils.meshing.volume_to_mesh, unmodified) and shown as its own napari
Surface layer alongside the 2D one, opt-in since mesh generation is real
compute next to just loading masks (see
docs/superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review
from sam2_utils import meshing, video_viz


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


def build_mesh_specs(root, neurons: list[str], *, preset: str = "faithful") -> list[dict]:
    """One mesh layer spec per neuron whose volume produced a real surface.

    Reuses render_review.neuron_volume (a second, independent call per neuron;
    not shared or cached with build_layer_specs's own call) and
    meshing.volume_to_mesh, both unmodified. volume_to_mesh returns vertices in
    (x, y, z) order for Blender; they are reversed back to (z, y, x) here to stay
    consistent with every other array in this module. Vertices already carry
    real-world nanometres (volume_to_mesh bakes `spacing` in), so only
    `translate` (origin * spacing, the same formula build_layer_specs uses) is
    needed, no `scale`. Each neuron's mesh gets a flat color from this repo's
    existing per-object palette (video_viz._color_for), indexed by the same
    1-based selection order build_layer_specs uses for its label IDs, so a
    neuron's 2D and 3D layers are not forced to match exactly but are both
    individually distinct from every other neuron's.
    """
    specs = []
    for i, neuron in enumerate(neurons, start=1):
        vol, origin, spacing = render_review.neuron_volume(root, neuron)
        if not vol.size or not vol.any():
            print(f"[view] {neuron}: no masks, skipping mesh")
            continue
        verts_xyz, faces = meshing.volume_to_mesh(vol, spacing=spacing, preset=preset)
        if len(faces) == 0:
            print(f"[view] {neuron}: no surface produced, skipping mesh")
            continue
        verts_zyx = verts_xyz[:, ::-1]
        translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))
        color = video_viz._color_for(i)
        vertex_colors = np.tile(color, (len(verts_zyx), 1))
        specs.append({"name": f"{neuron} mesh", "vertices": verts_zyx, "faces": faces,
                      "vertex_colors": vertex_colors, "translate": translate})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D/3D mask viewer (napari)")
    ap.add_argument("--tree", required=True,
                    help="bundle folder or output tree root (same layout either way)")
    ap.add_argument("--neurons", required=True,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL")
    ap.add_argument("--mesh", action="store_true",
                    help="also build and show each neuron's 3D mesh (slower: marching "
                         "cubes + smoothing per neuron, not built unless asked for)")
    args = ap.parse_args()

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    specs = build_layer_specs(args.tree, neurons)
    mesh_specs = build_mesh_specs(args.tree, neurons) if args.mesh else []
    if not specs and not mesh_specs:
        raise SystemExit(
            f"[view] none of {neurons} have any masks under {args.tree}")

    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    for spec in mesh_specs:
        viewer.add_surface((spec["vertices"], spec["faces"]), name=spec["name"],
                           vertex_colors=spec["vertex_colors"], translate=spec["translate"])
    opened = [s["name"] for s in specs] + [s["name"] for s in mesh_specs]
    print(f"[view] opened {len(opened)} layer(s): {', '.join(opened)}")
    napari.run()


if __name__ == "__main__":
    main()
