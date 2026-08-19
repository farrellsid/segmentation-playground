"""Build a self-contained review bundle from an output tree.

The bundle opens on a machine that has none of this project's data: no F:, no raw
EM store, no CATMAID cache. It carries identity (meta.json + neurons.csv), the
masks, per-chain QC, the frames to draw on, a manifest, and the slice of the
CATMAID tables the GUI needs (``data/chains.json`` + ``data/nodes.csv``), with
every ``state.json`` rewritten to point at a RELATIVE frames directory.

The ``data/`` slice is not optional. ``data/chains.json`` and
``data/aggregate_data_pv.csv`` are both gitignored, so a reviewer who clones the
repo has neither, and without them ``ReviewContext.find_chain`` and
``ReviewContext.annotate_df`` raise ``FileNotFoundError`` the moment she opens a
chain.

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


#: Legacy image-phase scale used when a chain's own state.json has none recorded.
_DEFAULT_SCALE = 8

#: Cache for the CATMAID context, keyed on the arguments it was built from so a
#: caller that loops over more than one tree (a library use, not just the CLI's
#: one-export-per-process) never gets back another tree's cfg or frames_root.
_CATMAID_CONTEXT: dict = {}


def _catmaid_context(output_root: Path, frames_root: Optional[Path]) -> tuple:
    """Return ``(chains_json, annotate_df, cfg, frames_root)``, loading once per key.

    Built lazily so an export whose frames all happen to exist locally never pays
    for the CATMAID node table, and so a test that stubs :func:`regenerate_frames`
    never touches it at all. Cached per ``(output_root, frames_root)`` pair rather
    than once per process: the CLI only ever calls this with one pair, so the
    cache still pays for itself there, but a notebook or a loop over several
    output trees now gets a ``cfg`` and ``frames_root`` that actually match the
    arguments it passed, instead of silently reusing whatever the first call saw.
    """
    key = (Path(output_root), Path(frames_root) if frames_root else None)
    if key not in _CATMAID_CONTEXT:
        import pandas as pd

        import pipeline
        from sam2_utils import alignment, config
        df = pd.read_csv(config.CSV_PATH)
        xy = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
        df["x_tif"], df["y_tif"] = xy[:, 0], xy[:, 1]
        _CATMAID_CONTEXT[key] = {
            "chains_json": json.loads(Path(config.CHAINS_PATH).read_text(encoding="utf-8")),
            "annotate_df": df,
            "cfg": pipeline.PipelineConfig(
                model_size="large", scale=_DEFAULT_SCALE, save_downscale=_DEFAULT_SCALE,
                output_root=output_root, frames_root=config.FRAMES_ROOT),
            "frames_root": Path(frames_root) if frames_root else Path(config.FRAMES_ROOT),
        }
    ctx = _CATMAID_CONTEXT[key]
    return ctx["chains_json"], ctx["annotate_df"], ctx["cfg"], ctx["frames_root"]


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

    chains_json, annotate_df, _cfg, froot = _catmaid_context(output_root, frames_root)
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
        # The chain's OWN recorded scale, not the freshly-fabricated cfg's: a chain
        # propagated at a non-default scale must regenerate at that same scale, or
        # the reviewer's canvas would not match the masks drawn on it.
        chain_cfg = state.get("config") or {}
        scale = int(chain_cfg.get("scale") or _DEFAULT_SCALE)
        frames_dir, _f2z, _a, _n = pipeline.prepare_video_frames(
            chain, annotate_df, scale=scale, frames_root=froot,
            anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
    return Path(frames_dir)


def _frames_complete(frames_dir: Path, state: dict) -> bool:
    """Whether ``frames_dir`` already holds a COMPLETE set of this chain's frames.

    Presence of ``00000.jpg`` alone is not proof of completeness: an export killed
    mid ``copytree`` (the exact ``F:`` drop-out this resumability exists for) can
    leave frame 0 on disk with everything after it missing, and the next run must
    not mistake that for done. ``state["n_frames"]`` is stamped by
    ``prepare_video_frames`` / ``prepare_chain_crop_frames`` when a chain is
    produced, so it is present for every real chain and is exactly the number of
    frames that should be on disk; comparing against it turns the resume check
    into a real completeness check instead of a presence check.

    Parameters
    ----------
    frames_dir : Path
        Candidate ``.../chain_XX/frames`` directory on the bundle side.
    state : dict
        The chain's parsed state.json.

    Returns
    -------
    bool
        True only when frame 0 exists and the on-disk JPEG count meets the
        recorded ``n_frames``. Falls back to presence-only when ``n_frames`` is
        not recorded (an older or hand-built state.json), since there is then
        nothing to verify a count against.
    """
    if not (frames_dir / "00000.jpg").exists():
        return False
    expected = state.get("n_frames")
    if not expected:
        return True
    return len(list(frames_dir.glob("*.jpg"))) >= int(expected)


def _replace_frames_dir(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst``, clearing ``dst`` first so stale files cannot survive.

    ``shutil.copytree(..., dirs_exist_ok=True)`` MERGES into an existing
    directory rather than replacing it. If ``dst`` already holds frames from a
    previous, incomplete or now-superseded write, and the new source has FEWER
    frames (a partial copy being resumed, or a re-regeneration under
    ``--force-frames``), leftover trailing files from the old write would stay on
    disk and silently inflate the frame count past what the new source actually
    has. Clearing first means ``dst`` always ends up matching ``src`` exactly.

    Parameters
    ----------
    src : Path
        Directory to copy frames from.
    dst : Path
        Destination directory, replaced wholesale.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


#: Columns the pipeline DERIVES from the raw table via alignment.catmaid_to_tif.
#: They are stripped before writing so the affine stays in code: baking x_tif /
#: y_tif into a bundle would freeze one version of the alignment into shipped
#: data, and a later correction to the affine could never reach it.
_DERIVED_NODE_COLUMNS = ("x_tif", "y_tif")


def _check_source_data_present() -> None:
    """Fail loudly when the CATMAID tables a bundle must carry are missing.

    Raises
    ------
    SystemExit
        When ``config.CSV_PATH`` or ``config.CHAINS_PATH`` is absent. Writing the
        bundle anyway would produce something that passes a casual look and then
        raises ``FileNotFoundError`` on the reviewer's machine, at the point where
        she can do the least about it.
    """
    from sam2_utils import config
    missing = [str(p) for p in (config.CHAINS_PATH, config.CSV_PATH) if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "cannot build a bundle without the CATMAID tables it must carry; missing: "
            + ", ".join(missing)
            + ". A bundle without data/chains.json and data/nodes.csv cannot be opened "
              "on a machine that has neither (both are gitignored).")


def write_bundle_data(dest: Path, neurons: List[str], *, output_root: Path,
                      frames_root: Optional[Path] = None) -> Path:
    """Write the CATMAID slice a bundle needs into ``<dest>/data``.

    Parameters
    ----------
    dest : Path
        The bundle root.
    neurons : list of str
        Cell names in this export. Both files are filtered to these NEURONS.
    output_root, frames_root : Path, Path optional
        Passed through to :func:`_catmaid_context`, whose already-loaded tables
        this reuses rather than re-reading them.

    Returns
    -------
    Path
        The ``data`` directory written.

    Notes
    -----
    Filtering is by NEURON, never by chain. ``chain_idx`` is a POSITION within a
    neuron's chain list, so dropping any of an exported neuron's chains would
    shift every later index for that neuron and the GUI would open the wrong
    chain. Each exported neuron therefore keeps its chains complete and in the
    source file's order, and only whole unexported neurons are dropped.

    The node table ships its RAW columns. ``x_tif`` / ``y_tif`` are recomputed by
    ``alignment.catmaid_to_tif`` on read, so the coordinate affine stays in code
    instead of being baked into shipped data.
    """
    chains_json, annotate_df, _cfg, _froot = _catmaid_context(output_root, frames_root)
    wanted = set(neurons)

    data_dir = Path(dest) / bundle.BUNDLE_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    subset = [c for c in chains_json if c.get("cell_name") in wanted]
    (data_dir / bundle.BUNDLE_CHAINS_NAME).write_text(json.dumps(subset, indent=2), encoding="utf-8")

    rows = annotate_df[annotate_df["cell_name"].isin(wanted)]
    drop = [c for c in _DERIVED_NODE_COLUMNS if c in rows.columns]
    rows.drop(columns=drop).to_csv(data_dir / bundle.BUNDLE_NODES_NAME, index=False)

    print(f"[export] data slice: {len(subset)} chain record(s), {len(rows)} node row(s)")
    return data_dir


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
    frames_root : Path, optional
        Scratch root regenerated frames are written under. Defaults to
        ``config.FRAMES_ROOT``. Only read for chains that need regeneration.
    force : bool, optional
        Allow writing into a non-empty destination.
    force_frames : bool, optional
        Regenerate (or re-copy) frames even for chains the bundle already has a
        complete frame set for.

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
    _check_source_data_present()

    dest.mkdir(parents=True, exist_ok=True)
    for rec in chains:
        src_dir = output_root / rec["chain_dir"]
        out_dir = dest / rec["chain_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copytree(src_dir / "masks", out_dir / "masks", dirs_exist_ok=True)
        if (src_dir / "qc.csv").exists():
            shutil.copy2(src_dir / "qc.csv", out_dir / "qc.csv")

        frames_out = out_dir / "frames"
        if _frames_complete(frames_out, rec["state"]) and not force_frames:
            print(f"[export] {rec['chain_dir']}: frames already present, skipping")
        else:
            # Frames are ALWAYS regenerated, never copied from the recorded
            # frames_dir. That directory cannot be trusted, for two independent
            # reasons found on real data. It is usually a dead Compute Canada
            # Narval /localscratch path, deleted when the job ended. And when it
            # does still exist, a `gui.py --anchor-only --context-frames N` review
            # session will have rewritten it to hold only the anchor plus or minus
            # N: the first real export found every one of 18 AIYL chains left with
            # exactly 5 frames, for chains needing anywhere from 1 to 43.
            #
            # A bare directory of numbered jpgs carries nothing that proves which
            # z each frame is, so a count cannot tell a complete set from a
            # narrowed one that happens to be the same size. Regenerating is the
            # only option correct by construction, and a reviewer checking masks
            # against the wrong EM slices is worse than one waiting a bit longer.
            print(f"[export] {rec['chain_dir']}: regenerating frames from raw EM")
            src = regenerate_frames(rec["state"], neuron=rec["cell_name"],
                                    chain_idx=rec["chain_idx"], output_root=output_root,
                                    frames_root=frames_root)
            _replace_frames_dir(src, frames_out)

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
    write_bundle_data(dest, names, output_root=output_root, frames_root=frames_root)
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
