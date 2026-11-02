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

With --show-stack, a third layer is added: the raw EM underneath the masks, cropped to
the selection's own bounding box and downscaled to the same 8x grid the mask layers
already use (see build_em_stack_spec), so a disjointed neuron is visible against real
anatomy instead of floating labels on nothing. Off by default, since it needs raw EM
access and real crop time; never cached into a --save-package (that travels to other
machines, this does not).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --show-stack
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --save-package <pkg dir>
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh --save-package <pkg dir>
    py -3 view_neurons.py --package <pkg dir>
    py -3 view_neurons.py --package <pkg dir> --mesh
"""
from __future__ import annotations

import argparse
import json
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review
from pipeline.frames import TifFrameStore, _read_tif_window
from sam2_utils import meshing, video_viz

#: _sam-grid downscale: raw EM pixels per _sam voxel in x/y (z is 1:1, no downscale).
#: Matches render_review.SPACING_SAM8_NM and every mask layer's own grid, so the EM
#: stack lines up with them with no separate alignment math.
_SAM_SCALE = 8


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


def build_em_stack_spec(specs: list[dict]) -> dict | None:
    """Raw EM, cropped to the union bounding box of `specs` and downscaled to the
    same _sam scale-8 grid the mask layers already use (render_review.SPACING_SAM8_NM),
    so it lines up with them with no separate alignment math: same spacing, same
    per-voxel grid, just a third layer underneath.

    This is for a quick look at whether a neuron's segmentation stays continuous
    across z, not pixel-perfect detail: an 8x downscale from native EM resolution,
    the same coarseness every mask layer already has, not the finer tier-2 crop a
    single chain's own review session loads. Cropping a 100% is the "untenable"
    cost that made this optional in the first place (see the repropagation-round-
    protocol doc for the same reasoning on the reviewer side).

    Returns None if `specs` is empty, there is nothing to bound.
    """
    if not specs:
        return None
    spacing = render_review.SPACING_SAM8_NM
    origins = [tuple(t / s for t, s in zip(spec["translate"], spec["scale"]))
              for spec in specs]
    z_lo = min(int(round(o[0])) for o in origins)
    z_hi = max(int(round(o[0])) + spec["data"].shape[0]
              for o, spec in zip(origins, specs))
    y_lo = min(int(round(o[1])) for o in origins)
    y_hi = max(int(round(o[1])) + spec["data"].shape[1]
              for o, spec in zip(origins, specs))
    x_lo = min(int(round(o[2])) for o in origins)
    x_hi = max(int(round(o[2])) + spec["data"].shape[2]
              for o, spec in zip(origins, specs))

    import cv2
    store = TifFrameStore()
    files = store.files_in_z_range(z_lo, z_hi - 1)
    path_by_z = {store.z_of_key(k): p for k, p in files}
    ny0, ny1 = y_lo * _SAM_SCALE, y_hi * _SAM_SCALE
    nx0, nx1 = x_lo * _SAM_SCALE, x_hi * _SAM_SCALE
    out_h, out_w = y_hi - y_lo, x_hi - x_lo
    sl = (slice(ny0, ny1), slice(nx0, nx1))

    planes = []
    missing = 0
    for z in range(z_lo, z_hi):
        path = path_by_z.get(z)
        if path is None:
            missing += 1
            planes.append(np.zeros((out_h, out_w), dtype=np.uint8))
            continue
        win = _read_tif_window(path, sl)
        gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
        planes.append(cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA))
    if missing:
        print(f"[view] EM stack: {missing}/{len(planes)} z-slice(s) had no tif, left blank")
    em_vol = np.stack(planes, axis=0)
    translate = (float(z_lo) * spacing[0], float(y_lo) * spacing[1], float(x_lo) * spacing[2])
    print(f"[view] EM stack: {em_vol.shape} (z, y, x) at {_SAM_SCALE}x downscale")
    return {"name": "EM (quick, downscaled)", "data": em_vol, "translate": translate,
            "scale": spacing}


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
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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


def save_em_stack(em_spec: dict, out_dir) -> None:
    """Write `em_spec` (the shape build_em_stack_spec() returns) into an existing
    package directory: the volume as one compressed .npy, plus an "em_stack" entry
    in manifest.json recording translate/scale/shape so load_em_stack_spec can
    rebuild the dict without guessing.

    Call this on an `out_dir` save_package() already wrote (manifest.json must
    exist); it only adds to that manifest, never creates a package on its own.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    np.save(out_dir / "em_stack.npy", em_spec["data"])
    manifest["em_stack"] = {
        "translate": [float(t) for t in em_spec["translate"]],
        "scale": [float(s) for s in em_spec["scale"]],
        "shape": list(em_spec["data"].shape),
        "file": "em_stack.npy",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_em_stack_spec(package_dir) -> dict | None:
    """The shape build_em_stack_spec() returns, read back from a package
    save_em_stack() wrote into. None if this package has no cached EM stack yet
    (show_stack was never requested for it, or it predates this feature)."""
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    entry = manifest.get("em_stack")
    if not entry:
        return None
    data = np.load(package_dir / entry["file"])
    return {"name": "EM (quick, downscaled)", "data": data,
            "translate": tuple(entry["translate"]), "scale": tuple(entry["scale"])}


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


def _build_from_tree(tree, neurons: list[str], *, mesh: bool = False):
    """Build layer specs (and, if requested, mesh specs) from a live tree/bundle.

    Shared by run() and main()'s --tree path, so the "no masks anywhere" check and the
    build order live in exactly one place.
    """
    specs = build_layer_specs(tree, neurons)
    mesh_specs = build_mesh_specs(tree, neurons) if mesh else []
    if not specs and not mesh_specs:
        raise SystemExit(f"[view] none of {neurons} have any masks under {tree}")
    return specs, mesh_specs


def _open_viewer(specs: list[dict], mesh_specs: list[dict], *,
                 show_stack: bool = False, em_spec: dict | None = None) -> None:
    """Open the napari window with the given layer specs.

    Shared by run() and both branches of main() (--tree and --package), so there is
    exactly one place that builds the window.

    `em_spec`, when given, is used directly (the launcher's cached path, see
    _run_cached: already loaded from or saved into the package, so this function
    does not rebuild it). Without one, `show_stack=True` builds it live from
    `specs`' own bounding box, the uncached main()/run() path: raw EM access is a
    local-machine thing, not something to bake into a package that might travel to
    a reviewer's machine with a different WORM_PATH, so --save-package never writes
    one either.
    """
    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    em_name = None
    if em_spec is None and show_stack:
        em_spec = build_em_stack_spec(specs)
    if em_spec is not None:
        viewer.add_image(em_spec["data"], name=em_spec["name"],
                         translate=em_spec["translate"], scale=em_spec["scale"])
        em_name = em_spec["name"]
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    for spec in mesh_specs:
        viewer.add_surface((spec["vertices"], spec["faces"]), name=spec["name"],
                           vertex_colors=spec["vertex_colors"], translate=spec["translate"])
    opened = ([em_name] if em_name else []) + [s["name"] for s in specs] + \
             [s["name"] for s in mesh_specs]
    print(f"[view] opened {len(opened)} layer(s): {', '.join(opened)}")
    napari.run()


def _cache_is_stale(source, neurons: list[str], saved_at: str) -> bool:
    """Whether any of `neurons`' chains were corrected (through gui.py) after this
    cache was saved.

    Reads `source`'s _review.csv directly (not via sam2_utils.review_queue, to avoid
    that import for one column read). An absent file or no "corrected" row for these
    neurons means no evidence anything changed, so the cache is trusted, not treated
    as stale by default. Only a "corrected" status counts: "approved"/"rejected" do
    not change mask pixels.
    """
    review_path = Path(source) / "_review.csv"
    if not review_path.exists():
        return False
    saved_dt = datetime.fromisoformat(saved_at)
    with review_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("neuron") in neurons and row.get("review_status") == "corrected"
                    and datetime.fromisoformat(row["updated_at"]) > saved_dt):
                return True
    return False


def _run_cached(source, neurons: list[str], *, mesh: bool, force_rebuild: bool,
                show_stack: bool = False) -> None:
    """The cache=True path run() delegates to: reuse <source>/.view_package/ when it
    exists, was built from exactly this neuron set, is not stale, and (if mesh is
    requested) already has mesh data; otherwise build live and overwrite the cache so
    the NEXT call is fast.

    show_stack rides the same reuse/rebuild decision as the masks: when the cached
    package is being reused, its em_stack entry (if any) is reused too; otherwise a
    fresh EM crop is built and saved into the (possibly freshly rebuilt) package, so
    the NEXT call with show_stack=True is instant instead of re-cropping every time
    (the actual gripe this cache exists to fix). A reuse call with show_stack=True
    against a package saved WITHOUT one still builds and saves it now, on top of the
    otherwise-reused masks, rather than waiting for a mask rebuild to earn it.
    """
    cache_dir = Path(source) / ".view_package"
    manifest_path = cache_dir / "manifest.json"
    if not force_rebuild and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        same_neurons = sorted(manifest["built_from"]["neurons"]) == sorted(neurons)
        has_mesh = bool(manifest.get("meshes"))
        fresh = not _cache_is_stale(source, neurons, manifest["saved_at"])
        if same_neurons and fresh and (not mesh or has_mesh):
            specs = load_package(cache_dir)
            mesh_specs = load_mesh_specs(cache_dir) if mesh else []
            em_spec = None
            if show_stack:
                em_spec = load_em_stack_spec(cache_dir)
                if em_spec is None:
                    em_spec = build_em_stack_spec(specs)
                    if em_spec is not None:
                        save_em_stack(em_spec, cache_dir)
            _open_viewer(specs, mesh_specs, em_spec=em_spec)
            return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    save_package(specs, cache_dir, built_from={"tree": str(source), "neurons": neurons},
                mesh_specs=mesh_specs if mesh else None)
    em_spec = None
    if show_stack:
        em_spec = build_em_stack_spec(specs)
        if em_spec is not None:
            save_em_stack(em_spec, cache_dir)
    _open_viewer(specs, mesh_specs, em_spec=em_spec)


def run(source, neurons: list[str], *, mesh: bool = False,
       cache: bool = False, force_rebuild: bool = False,
       show_stack: bool = False) -> None:
    """Build and open the viewer directly, the same shape render_review.run(source,
    neurons) already has (the function launcher.py's "Render video + mesh" button
    calls). This is what launcher.py's "View neurons" button calls.

    cache=True (launcher-only; main()'s --tree CLI path never sets it) reuses a saved
    package at <source>/.view_package/ instead of always rebuilding live; see
    _run_cached. force_rebuild=True skips the cache and staleness check entirely.

    show_stack adds a quick, downscaled raw-EM layer underneath the masks (see
    build_em_stack_spec), for spotting where a neuron goes disjointed across z. Off
    by default: it needs raw EM access (WORM_PATH) and costs real time to crop, the
    same cost that makes shipping frames to a reviewer worth doing centrally instead
    (see docs/how-to/repropagation-round-protocol.md).

    --save-package/--package are CLI-only concerns this entry point does not take;
    see main() below for those.
    """
    if cache:
        _run_cached(source, neurons, mesh=mesh, force_rebuild=force_rebuild,
                   show_stack=show_stack)
        return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    _open_viewer(specs, mesh_specs, show_stack=show_stack)


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
    ap.add_argument("--show-stack", action="store_true",
                    help="also show the raw EM underneath the masks, cropped to the "
                         "selection's bounding box and downscaled 8x (same grid the "
                         "masks already use), for spotting where a neuron goes "
                         "disjointed across z. Needs raw EM access (WORM_PATH) and "
                         "costs real time to crop, so it's opt-in.")
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
        specs, mesh_specs = _build_from_tree(args.tree, neurons, mesh=args.mesh)
        if args.save_package:
            save_package(specs, args.save_package,
                         built_from={"tree": args.tree, "neurons": neurons},
                         mesh_specs=mesh_specs)
            print(f"[view] saved package to {args.save_package}")

    _open_viewer(specs, mesh_specs, show_stack=args.show_stack)


if __name__ == "__main__":
    main()
