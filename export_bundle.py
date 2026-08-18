"""Build a self-contained review bundle from an output tree.

The bundle opens on a machine that has none of this project's data: no F:, no raw
EM store, no CATMAID cache. It carries identity (meta.json + neurons.csv), the
masks, per-chain QC, the frames to draw on, and a manifest, with every
``state.json`` rewritten to point at a RELATIVE frames directory.

    py -3 export_bundle.py --output-root "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed" \\
        --dest "F:\\ZhenLab\\Data\\bundles\\AIA_for_lucinda" --neurons AIAL AIAR

Zip the result and send it. She returns it, and import_bundle.py merges her
corrections back.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from sam2_utils import bundle, chain_meta, registry


def _find_chain(chains_json: list, neuron: str, chain_idx: int) -> dict:
    """The chain dict for (neuron, chain_idx).

    ``chain_idx`` is a POSITION within that neuron's chain list, not an id, matching
    ``gui.ReviewContext.find_chain`` and ``batch.enumerate_chains``.
    """
    chs = [c for c in chains_json if c.get("cell_name") == neuron]
    if not 0 <= chain_idx < len(chs):
        raise SystemExit(f"{neuron} chain_{chain_idx:02d} is not in chains.json "
                         f"({len(chs)} chain(s) for that neuron)")
    return chs[chain_idx]


#: Cache for the CATMAID context, built at most once per process.
_CATMAID_CONTEXT: dict = {}


def _catmaid_context(output_root: Path, frames_root: Optional[Path]) -> tuple:
    """Return ``(chains_json, annotate_df, cfg, frames_root)``, loading once.

    Built lazily so an export whose frames all happen to exist locally never pays
    for the CATMAID node table, and so a test that stubs :func:`regenerate_frames`
    never touches it at all.
    """
    if not _CATMAID_CONTEXT:
        import pandas as pd

        import pipeline
        from sam2_utils import alignment, config
        df = pd.read_csv(config.CSV_PATH)
        xy = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
        df["x_tif"], df["y_tif"] = xy[:, 0], xy[:, 1]
        _CATMAID_CONTEXT["chains_json"] = json.loads(
            Path(config.CHAINS_PATH).read_text(encoding="utf-8"))
        _CATMAID_CONTEXT["annotate_df"] = df
        _CATMAID_CONTEXT["cfg"] = pipeline.PipelineConfig(
            model_size="large", scale=8, save_downscale=8,
            output_root=output_root, frames_root=config.FRAMES_ROOT)
        _CATMAID_CONTEXT["frames_root"] = (Path(frames_root) if frames_root
                                           else Path(config.FRAMES_ROOT))
    return (_CATMAID_CONTEXT["chains_json"], _CATMAID_CONTEXT["annotate_df"],
            _CATMAID_CONTEXT["cfg"], _CATMAID_CONTEXT["frames_root"])


def regenerate_frames(state: dict, *, neuron: str, chain_idx: int,
                      output_root: Path, frames_root: Optional[Path] = None) -> Path:
    """Rebuild a chain's frames from the raw EM store and return the directory.

    A chain's recorded ``frames_dir`` is an absolute path baked in at generation
    time, and for every chain produced on the cluster it is a Narval
    ``/localscratch/<jobid>/...`` directory that stopped existing when the job
    ended. Regenerating is therefore the normal path for an export, not a fallback,
    and it is the reason export needs the raw EM store and ``F:`` mounted.

    Parameters
    ----------
    state : dict
        The chain's parsed state.json. ``crop_window`` decides which space is built.
    neuron, chain_idx : str, int
        Identify the chain and namespace the generated view directory. ``chain_idx``
        is a POSITION within that neuron's chain list, not an id.
    output_root : Path
        The tree being exported, used only to build a ``PipelineConfig``.
    frames_root : Path, optional
        Scratch root the frames are written under. Defaults to ``config.FRAMES_ROOT``.

    Returns
    -------
    Path
        Directory holding the 0-indexed JPEG frames.
    """
    import pipeline
    from sam2_utils import alignment

    chains_json, annotate_df, cfg, froot = _catmaid_context(output_root, frames_root)
    chain = _find_chain(chains_json, neuron, chain_idx)
    anchor_z = int(state["anchor_catmaid_z"])
    cw_dict = state.get("crop_window")
    if cw_dict:
        cw = alignment.CropWindow(
            origin_tif=tuple(cw_dict["origin_tif"]), size_tif=tuple(cw_dict["size_tif"]),
            crop_scale=int(cw_dict["crop_scale"]), sam_scale=int(cw_dict["sam_scale"]))
        frames_dir, _f2z, _a, _n = pipeline.prepare_chain_crop_frames(
            chain, annotate_df, cw, frames_root=froot, anchor_catmaid_z=anchor_z,
            neuron=neuron, chain_idx=chain_idx)
    else:
        frames_dir, _f2z, _a, _n = pipeline.prepare_video_frames(
            chain, annotate_df, scale=cfg.scale, frames_root=froot,
            anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
    return Path(frames_dir)


def export_bundle(output_root: Path, dest: Path, *, neurons: Optional[List[str]] = None,
                  source_tree: Optional[str] = None, backend: str = "sam2",
                  reprop_variant: Optional[str] = None, frames_root: Optional[Path] = None,
                  force: bool = False, force_frames: bool = False) -> dict:
    """Write a review bundle for ``neurons`` from ``output_root`` into ``dest``.

    Parameters
    ----------
    output_root : Path
        The master tree to export from. Only ever read.
    dest : Path
        Bundle directory to create. Must be absent or empty unless ``force``.
    neurons : list of str, optional
        Cell names to include. None exports every neuron in the tree.
    source_tree : str, optional
        Provenance label. Defaults to ``output_root.name``.
    backend : str, optional
        ``"sam2"`` or ``"sam3"``, recorded in each meta.json.
    reprop_variant : str, optional
        ``"mask_seed"`` or ``"box_seed"`` when exporting a re-propagation tree.
    force : bool, optional
        Allow writing into a non-empty destination.

    Returns
    -------
    dict
        The manifest that was written.

    Raises
    ------
    SystemExit
        If ``dest`` is non-empty and ``force`` is False. Overwriting a bundle that
        may already hold returned corrections would lose work silently.
    """
    output_root, dest = Path(output_root), Path(dest)
    if dest.exists() and any(dest.iterdir()) and not force:
        raise SystemExit(f"{dest} is not empty. Pass --force only if you are sure nothing "
                         f"there is a returned bundle with corrections in it.")

    reg = registry.load_registry()
    source_tree = source_tree or output_root.name
    chains = bundle.index_chains(output_root, neurons=neurons)
    if not chains:
        raise SystemExit(f"no chains found under {output_root} for neurons={neurons}")

    dest.mkdir(parents=True, exist_ok=True)
    for rec in chains:
        src_dir = output_root / rec["chain_dir"]
        out_dir = dest / rec["chain_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(src_dir / "masks", out_dir / "masks", dirs_exist_ok=True)
        if (src_dir / "qc.csv").exists():
            shutil.copy2(src_dir / "qc.csv", out_dir / "qc.csv")

        if (out_dir / "frames" / "00000.jpg").exists() and not force_frames:
            print(f"[export] {rec['chain_dir']}: frames already present, skipping")
        else:
            recorded = rec["state"].get("frames_dir")
            frames_src = Path(recorded) if recorded and not str(recorded).startswith("/") else None
            if frames_src is not None and frames_src.is_dir():
                shutil.copytree(frames_src, out_dir / "frames", dirs_exist_ok=True)
            else:
                # The recorded path is almost always a dead Narval /localscratch dir,
                # so regenerating from the raw EM store is the normal path, not a fallback.
                print(f"[export] {rec['chain_dir']}: regenerating frames from raw EM")
                src = regenerate_frames(rec["state"], neuron=rec["cell_name"],
                                        chain_idx=rec["chain_idx"], output_root=output_root,
                                        frames_root=frames_root)
                shutil.copytree(src, out_dir / "frames", dirs_exist_ok=True)

        state_out = bundle.rewrite_state_frames_dir(rec["state"])
        (out_dir / "state.json").write_text(json.dumps(state_out, indent=2), encoding="utf-8")

        nid = registry.neuron_id(rec["cell_name"], registry=reg)
        meta = chain_meta.build_meta(rec["state"], neuron_id=nid, source_tree=source_tree,
                                     backend=backend, reprop_variant=reprop_variant,
                                     chain_dir=out_dir)
        chain_meta.write_meta(out_dir, meta)

    manifest = bundle.build_manifest(chains, source_tree=source_tree)
    (dest / bundle.BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    names = sorted({c["cell_name"] for c in chains})
    with (dest / "neurons.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["neuron_id", "cell_name"])
        for name in names:
            w.writerow([registry.neuron_id(name, registry=reg), name])

    problems = bundle.validate_bundle(dest)
    if problems:
        print("[export] bundle validation problems:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
    else:
        print(f"[export] wrote {len(chains)} chain(s), {len(names)} neuron(s) to {dest}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--neurons", nargs="*", default=None)
    ap.add_argument("--source-tree", default=None)
    ap.add_argument("--backend", default="sam2", choices=["sam2", "sam3"])
    ap.add_argument("--reprop-variant", default=None, choices=["mask_seed", "box_seed"])
    ap.add_argument("--frames-root", type=Path, default=None,
                    help="scratch root for regenerated frames; defaults to config.FRAMES_ROOT")
    ap.add_argument("--force", action="store_true",
                    help="write into a non-empty destination (check it is not a returned bundle)")
    ap.add_argument("--force-frames", action="store_true",
                    help="regenerate frames even for chains the bundle already has them for")
    args = ap.parse_args()
    export_bundle(args.output_root, args.dest, neurons=args.neurons,
                  source_tree=args.source_tree, backend=args.backend,
                  reprop_variant=args.reprop_variant, frames_root=args.frames_root,
                  force=args.force, force_frames=args.force_frames)


if __name__ == "__main__":
    main()
