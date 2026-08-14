"""Assemble a .docx comparing mask-seed vs box-seed re-propagation, for the chains
whose anchor was hand-corrected and then re-propagated by
`cluster/run_reprop_corrected_seed.sh`. Reads the same `neuron,chain_idx,anchor_z`
manifest CSV that job consumed, so the table only covers chains that actually went
through reprop, not every chain in the neuron.

Column mapping (a deliberate departure from build_docx_report.py's anchor-frame
convention): every snapshot column shows the frame FARTHEST from the anchor within
its own tree, not the anchor frame. The anchor is identical across all three trees
by construction (mask-seed and box-seed both start from the same corrected seed), so
an anchor-frame snapshot would show no difference at all; the interesting question is
how far each variant's propagation drifted by the time it reaches the far end of the
chain, and that is exactly where mask-seed and box-seed can diverge from each other
and from the pre-reprop "before" tail.

Run (after render_reprop_report.py has already produced
<assets>/<neuron>/chain_NN_{before,mask,box}.gif for every manifest row):

    py -3 experiments/build_reprop_docx_report.py \\
        --before "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIAL_AIAR" \\
        --mask-tree "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed_AIA" \\
        --box-tree "F:\\ZhenLab\\Data\\output_masks\\reprop_boxseed_AIA" \\
        --manifest cluster/corrected_chains_AIA.csv \\
        --assets "F:\\ZhenLab\\Data\\repo_offload\\report_assets\\reprop_AIA" \\
        --out "F:\\ZhenLab\\Data\\repo_offload\\report_assets\\reprop_AIA\\AIA_reprop_report.docx"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from docx import Document
from docx.shared import Inches

import pipeline

SIDES = ("before", "mask", "box")
SIDE_LABELS = {"before": "Before", "mask": "Mask-seed Reprop", "box": "Box-seed Reprop"}


def _read_manifest(path: Path) -> list[tuple[str, int, int]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["neuron"], int(row["chain_idx"]), int(row["anchor_z"])))
    return rows


def _farthest_frame_index(tree: Path, neuron: str, chain_idx: int, anchor_z: int) -> int:
    """Index, within this chain's own sorted z-list IN THIS TREE, of the z farthest
    from the anchor. Computed per tree (not shared across before/mask/box) because
    the three variants can have different tail lengths, e.g. one dropping out
    earlier than another."""
    cdir = tree / neuron / f"chain_{chain_idx:02d}"
    masks = pipeline.chain_masks_in_sam(cdir)
    zs = sorted(masks.keys())
    if not zs:
        return 0
    return max(range(len(zs)), key=lambda i: abs(zs[i] - anchor_z))


def extract_farthest_png(assets: Path, frames_out: Path, tree: Path, neuron: str,
                         chain_idx: int, anchor_z: int, side: str) -> Path:
    gif_path = assets / neuron / f"chain_{chain_idx:02d}_{side}.gif"
    out_path = frames_out / f"{neuron}_chain_{chain_idx:02d}_{side}_far.png"
    idx = _farthest_frame_index(tree, neuron, chain_idx, anchor_z)
    im = Image.open(gif_path)
    idx = min(idx, im.n_frames - 1)
    im.seek(idx)
    im.convert("RGB").save(out_path)
    return out_path


def build_table(doc: Document, before: Path, mask_tree: Path, box_tree: Path,
                assets: Path, frames_out: Path, neuron: str,
                rows: list[tuple[int, int]]) -> list[int]:
    """rows: [(chain_idx, anchor_z), ...] for this neuron only."""
    trees = {"before": before, "mask": mask_tree, "box": box_tree}
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    headers = ["Chain",
              "Before (Farthest Frame)",
              "Mask-seed Reprop (Farthest Frame)",
              "Box-seed Reprop (Farthest Frame)",
              "Renders",
              "Note"]
    for i, text in enumerate(headers):
        hdr[i].text = text

    missing = []
    for chain_idx, anchor_z in rows:
        gif_paths = {s: assets / neuron / f"chain_{chain_idx:02d}_{s}.gif" for s in SIDES}
        if not all(p.exists() for p in gif_paths.values()):
            row = table.add_row().cells
            row[0].text = str(chain_idx)
            row[5].text = "Not yet rendered (run render_reprop_report.py for this chain)"
            missing.append(chain_idx)
            print(f"  {neuron} chain_{chain_idx:02d}: gif(s) missing, skipped for now")
            continue

        row = table.add_row().cells
        row[0].text = str(chain_idx)
        for col, side in zip((1, 2, 3), SIDES):
            png = extract_farthest_png(assets, frames_out, trees[side], neuron,
                                       chain_idx, anchor_z, side)
            row[col].paragraphs[0].add_run().add_picture(str(png), width=Inches(1.3))
        render_lines = [f"{SIDE_LABELS[s]}: {neuron}/chain_{chain_idx:02d}_{s}.gif"
                        for s in SIDES]
        row[4].text = "\n".join(render_lines)
        print(f"  {neuron} chain_{chain_idx:02d}: rendered")
    return missing


def add_merged_render_section(doc: Document, assets: Path, frames_out: Path,
                              neuron: str) -> bool:
    """Whole-neuron merged render (every reprop'd chain overlaid in one crop), from
    report_assets.py's neuron-gif-triple, one MIDDLE frame per variant as an
    orientation thumbnail. Unlike the per-chain table's farthest-from-anchor
    convention, a merged multi-chain view has no single "anchor" or "far end" to
    key off, all its chains have their own separate anchors and lengths, so the
    middle frame of the merged z-range is just a reasonable orientation snapshot,
    not a claim about where the variants differ most. Returns False (and adds
    nothing) if `render_reprop_report.py`'s neuron-gif-triple output isn't there
    yet for this neuron, expected under <assets>/<neuron>_merged_{side}.gif."""
    gif_paths = {s: assets / f"{neuron}_merged_{s}.gif" for s in SIDES}
    if not all(p.exists() for p in gif_paths.values()):
        return False

    doc.add_paragraph("Merged Render (all reprop'd chains, one crop)").bold = True
    table = doc.add_table(rows=2, cols=3)
    table.style = "Table Grid"
    for col, side in zip(range(3), SIDES):
        table.rows[0].cells[col].text = SIDE_LABELS[side]
        im = Image.open(gif_paths[side])
        im.seek(im.n_frames // 2)
        png_path = frames_out / f"{neuron}_merged_{side}_mid.png"
        im.convert("RGB").save(png_path)
        table.rows[1].cells[col].paragraphs[0].add_run().add_picture(
            str(png_path), width=Inches(1.8))
    render_lines = [f"{SIDE_LABELS[s]}: {neuron}_merged_{s}.gif" for s in SIDES]
    doc.add_paragraph("\n".join(render_lines)).italic = True
    return True


def build_report(before: Path, mask_tree: Path, box_tree: Path, manifest: Path,
                 assets: Path, out_path: Path) -> None:
    frames_out = assets / "anchor_frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest(manifest)
    by_neuron: dict[str, list[tuple[int, int]]] = {}
    for neuron, chain_idx, anchor_z in manifest_rows:
        by_neuron.setdefault(neuron, []).append((chain_idx, anchor_z))
    for rows in by_neuron.values():
        rows.sort()

    doc = Document()
    doc.add_heading("Re-propagation Comparison Report", level=0)
    note = doc.add_paragraph()
    note.add_run(
        "Covers only the chains whose anchor was hand-corrected and then "
        "re-propagated (this manifest's rows), not every chain in the neuron. Each "
        "snapshot column shows the frame FARTHEST from the anchor within its own "
        "tree, not the anchor frame itself, since the anchor is identical across "
        "all three trees by construction and the interesting differences show up "
        "at the far end of the propagated tail. Render columns name the animated "
        "gif for that chain instead of embedding it, since neither Word nor Google "
        "Docs plays animated GIFs; the real gif files are included alongside this "
        "document (same folder/zip you got this from), under <neuron>\\, if you want "
        "to see the actual motion. Each neuron section opens with a whole-neuron "
        "Merged Render (every reprop'd chain overlaid in one crop, middle-frame "
        "snapshot) when that gif has been rendered, before the per-chain table."
    ).italic = True

    all_missing: dict[str, list[int]] = {}
    for neuron, rows in by_neuron.items():
        doc.add_heading(neuron, level=1)
        print(f"=== {neuron} ===")
        if add_merged_render_section(doc, assets, frames_out, neuron):
            print(f"  {neuron}: merged render added")
        else:
            print(f"  {neuron}: no merged render yet (neuron-gif-triple not run), skipped")
        missing = build_table(doc, before, mask_tree, box_tree, assets, frames_out,
                              neuron, rows)
        if missing:
            all_missing[neuron] = missing
        doc.add_page_break()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"wrote {out_path}")
    if all_missing:
        print("[build-reprop-docx] some manifest chains had no rendered gif yet, noted in "
             "the doc instead of embedding an image, run render_reprop_report.py for "
             "these and re-run this script:")
        for neuron, chains in all_missing.items():
            chain_list = ",".join(f"{c:02d}" for c in chains)
            print(f"  {neuron}: {chain_list}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="pre-reprop working tree")
    ap.add_argument("--mask-tree", required=True, help="mask-seed reprop output tree")
    ap.add_argument("--box-tree", required=True, help="box-seed reprop output tree")
    ap.add_argument("--manifest", required=True, help="neuron,chain_idx,anchor_z CSV")
    ap.add_argument("--assets", required=True,
                    help="render_reprop_report.py output root (has <neuron>/chain_NN_{before,mask,box}.gif)")
    ap.add_argument("--out", required=True, help="output .docx path")
    args = ap.parse_args(argv)

    build_report(Path(args.before), Path(args.mask_tree), Path(args.box_tree),
                Path(args.manifest), Path(args.assets), Path(args.out))


if __name__ == "__main__":
    main()
