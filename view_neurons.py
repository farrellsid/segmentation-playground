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

With --save-package, the built 2D layers are also saved to disk as a VAST-style
labelmap package (one 16-bit PNG per z-slice plus manifest.json, this repo's
existing VAST-GT convention from eval/groundtruth.py, not a new format), so a
later run can reopen it with --package instead of rebuilding from --tree
(see docs/superpowers/specs/2026-10-04-neuron-package-design.md). Combined with
--mesh, each neuron's mesh is also cached into the same package (one .ply per
neuron plus a "meshes" entry in manifest.json, see
docs/superpowers/specs/2026-10-04-neuron-mesh-package-design.md); --package
--mesh then loads those meshes back instead of recomputing them, and fails
clearly if the package was saved without --mesh.

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --save-package <pkg dir>
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh --save-package <pkg dir>
    py -3 view_neurons.py --package <pkg dir>
    py -3 view_neurons.py --package <pkg dir> --mesh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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


def save_package(specs: list[dict], out_dir, *, built_from: dict,
                 mesh_specs: list[dict] | None = None) -> Path:
    """Write `specs` (the same shape build_layer_specs() returns) to `out_dir`
    as a VAST-style labelmap package: one 16-bit single-channel PNG per
    z-slice (pixel value is the neuron ID, 0 is background, the same
    convention eval/groundtruth.py already reads for VAST GT data), plus
    manifest.json.

    Reuses each spec's already-computed id-multiplied `data`, `translate`, and
    `scale` directly; never calls render_review.neuron_volume itself, no
    second pass over the source data. `built_from` is recorded in the
    manifest exactly as given, not interpreted here.

    Overlap policy: later entries in `specs` overwrite earlier ones only on
    the pixels the later entry itself claims as foreground (non-zero), not on
    every pixel inside its bounding box. A later neuron's own background
    pixels, the empty margin inside its own crop that is normal for an
    irregular cell shape, never erase an earlier neuron's real foreground
    sitting underneath that margin; only an actual overlap between the two
    neurons' own shapes resolves to last-wins. A dense single-valued array
    cannot show two neurons' claims on one pixel the way separate live layers
    blend visually; this masked last-wins rule is a real, deliberate
    simplification (see the design doc), not full overlap arbitration.

    `mesh_specs` (the same shape `build_mesh_specs()` returns), when given and non-empty, is
    also written: one `.ply` per neuron under `meshes/`, plus a `"meshes"` entry in the manifest.
    Omitted or empty, the package is 2D-only, exactly as before this parameter existed.
    """
    if not specs:
        raise SystemExit("[view] nothing to save, no neuron had any masks")
    out_dir = Path(out_dir)
    spacing = specs[0]["scale"]
    origins = [tuple(int(round(t / s)) for t, s in zip(spec["translate"], spacing))
              for spec in specs]
    z0 = min(o[0] for o in origins)
    y0 = min(o[1] for o in origins)
    x0 = min(o[2] for o in origins)
    z1 = max(o[0] + spec["data"].shape[0] for o, spec in zip(origins, specs))
    y1 = max(o[1] + spec["data"].shape[1] for o, spec in zip(origins, specs))
    x1 = max(o[2] + spec["data"].shape[2] for o, spec in zip(origins, specs))
    shape = (z1 - z0, y1 - y0, x1 - x0)

    canvas = np.zeros(shape, dtype=np.uint16)
    neuron_entries = []
    for spec, origin in zip(specs, origins):
        data = spec["data"]
        dz, dy, dx = origin[0] - z0, origin[1] - y0, origin[2] - x0
        sz, sy, sx = data.shape
        region = canvas[dz:dz + sz, dy:dy + sy, dx:dx + sx]
        mask = data > 0
        region[mask] = data[mask]
        neuron_entries.append({"name": spec["name"], "id": int(data.max())})

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(shape[0]):
        Image.fromarray(canvas[i].astype(np.uint16), mode="I;16").save(
            labels_dir / f"{i:06d}.png")

    manifest = {
        "schema_version": 1,
        "built_from": built_from,
        "origin": [z0, y0, x0],
        "spacing": list(spacing),
        "shape": list(shape),
        "neurons": neuron_entries,
    }

    if mesh_specs:
        id_by_name = {entry["name"]: entry["id"] for entry in neuron_entries}
        meshes_dir = out_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        mesh_entries = []
        for spec in mesh_specs:
            name = spec["name"].removesuffix(" mesh")
            if name not in id_by_name:
                raise SystemExit(
                    f"[view] mesh for {name!r} has no matching 2D neuron entry in this "
                    f"package; save_package needs mesh_specs built from the same neuron "
                    f"list as specs")
            verts_xyz = spec["vertices"][:, ::-1]
            ply_path = meshes_dir / f"{name}.ply"
            meshing.write_ply(ply_path, verts_xyz, spec["faces"])
            mesh_entries.append({
                "name": name,
                "id": id_by_name[name],
                "translate": [float(t) for t in spec["translate"]],
                "file": ply_path.name,
            })
        manifest["meshes"] = mesh_entries

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_package(package_dir) -> list[dict]:
    """The same layer-spec shape build_layer_specs() returns, read back from a
    package save_package() wrote.

    Every neuron's layer here is the full package shape, not re-cropped to
    that one neuron's own tighter bounding box the way the live --tree path
    crops each neuron individually; packages are already small (a handful of
    neurons), and re-deriving each neuron's own crop from the merged array is
    unneeded complexity for what it would save.
    """
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    spacing = tuple(float(s) for s in manifest["spacing"])
    origin = tuple(int(o) for o in manifest["origin"])
    shape = tuple(int(s) for s in manifest["shape"])
    translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))

    canvas = np.zeros(shape, dtype=np.uint16)
    for i in range(shape[0]):
        with Image.open(package_dir / "labels" / f"{i:06d}.png") as im:
            arr = np.asarray(im)
        canvas[i] = arr if arr.dtype == np.uint16 else arr.astype(np.uint16)

    specs = []
    for entry in manifest["neurons"]:
        neuron_id = int(entry["id"])
        data = np.where(canvas == neuron_id, np.uint16(neuron_id), np.uint16(0))
        specs.append({"name": entry["name"], "data": data, "translate": translate,
                      "scale": spacing})
    return specs


def load_mesh_specs(package_dir) -> list[dict]:
    """The same layer-spec shape build_mesh_specs() returns, read back from a package
    save_package() wrote with mesh_specs. Raises a clear SystemExit if the package was
    saved without --mesh (no "meshes" key in its manifest), rather than silently
    returning an empty list.
    """
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    mesh_entries = manifest.get("meshes")
    if not mesh_entries:
        raise SystemExit(
            f"[view] package at {package_dir} has no saved mesh data; re-save with "
            f"--tree ... --mesh --save-package to include meshes")

    specs = []
    for entry in mesh_entries:
        verts_xyz, faces = meshing.read_ply(package_dir / "meshes" / entry["file"])
        verts_zyx = verts_xyz[:, ::-1]
        translate = tuple(float(t) for t in entry["translate"])
        color = video_viz._color_for(entry["id"])
        vertex_colors = np.tile(color, (len(verts_zyx), 1))
        specs.append({"name": f"{entry['name']} mesh", "vertices": verts_zyx,
                      "faces": faces, "vertex_colors": vertex_colors,
                      "translate": translate})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D/3D mask viewer (napari)")
    ap.add_argument("--tree", default=None,
                    help="bundle folder or output tree root (same layout either way); "
                         "required together with --neurons, unless --package is given")
    ap.add_argument("--neurons", default=None,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL; required "
                         "together with --tree, unless --package is given")
    ap.add_argument("--mesh", action="store_true",
                    help="also build and show each neuron's 3D mesh (slower: marching "
                         "cubes + smoothing per neuron, not built unless asked for); "
                         "with --package, loads a saved mesh instead of recomputing it")
    ap.add_argument("--save-package", default=None, metavar="DIR",
                    help="also save the --tree/--neurons selection as a VAST-style "
                         "labelmap package to this directory; with --mesh, also caches "
                         "each neuron's mesh into the same package")
    ap.add_argument("--package", default=None, metavar="DIR",
                    help="load from a previously saved package instead of --tree/--neurons; "
                         "add --mesh to also load that package's cached meshes")
    args = ap.parse_args()

    using_tree = args.tree is not None or args.neurons is not None
    if args.package and using_tree:
        raise SystemExit("[view] --package cannot be combined with --tree/--neurons")
    if args.package and args.save_package:
        raise SystemExit("[view] --save-package only applies when building from "
                         "--tree/--neurons, not --package")
    if not args.package and not (args.tree and args.neurons):
        raise SystemExit("[view] need either --package, or both --tree and --neurons")

    mesh_specs = []
    if args.package:
        specs = load_package(args.package)
        if not specs:
            raise SystemExit(f"[view] package at {args.package} has no neurons")
        if args.mesh:
            mesh_specs = load_mesh_specs(args.package)
    else:
        neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
        specs = build_layer_specs(args.tree, neurons)
        if args.mesh:
            mesh_specs = build_mesh_specs(args.tree, neurons)
        if not specs and not mesh_specs:
            raise SystemExit(
                f"[view] none of {neurons} have any masks under {args.tree}")
        if args.save_package:
            save_package(specs, args.save_package,
                         built_from={"tree": args.tree, "neurons": neurons},
                         mesh_specs=mesh_specs)
            print(f"[view] saved package to {args.save_package}")

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
