"""Assemble a .docx Pipeline Report from report_assets.py's already-rendered chain
gifs, for pasting into Google Docs: markdown import does not bring GIFs in cleanly,
and neither Word nor Google Docs plays an animated GIF even when it does import, so
this builds a real docx with static images in the right template columns instead.

Column mapping (matches the "Pipeline Report" template):
    Before/After (Start Frame): a static PNG of the ANCHOR frame (the one actually
        corrected), one image per chain. For a corrected chain this is extracted
        from the already-rendered before/after gif (report_assets.py chain-gif-pair);
        for an uncorrected chain (the human reviewed it and left it as-is) it is
        rendered directly from the BEFORE tree, Before column only, since there is
        no "after" to show.
    Before/After (Render): the animated content belongs here conceptually, but
        cannot survive into docx/Google Docs, so these cells hold the gif's
        filename as a pointer back to the real file, not an image.

Uses experiments/find_corrected_chains.py's own find_corrected_chains() to determine
which chains actually differ between the two trees, rather than a hardcoded list, so
this works for any before/after tree pair, not just one specific run.

Run (after report_assets.py chain-gif-pair has already produced the gifs under
<assets>/<neuron>/chain_NN_{before,after}.gif for every corrected chain):

    py -3 experiments/build_docx_report.py \\
        --before "F:\\ZhenLab\\Data\\output_masks\\resolution_experiments\\target_perslice_only_guard_sam3_merged" \\
        --after  "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIAL_AIAR" \\
        --neurons AIAL,AIAR \\
        --assets "F:\\ZhenLab\\Data\\repo_offload\\report_assets" \\
        --out "F:\\ZhenLab\\Data\\repo_offload\\report_assets\\AIAL_AIAR_report.docx"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from docx import Document
from docx.shared import Inches

import pipeline
from sam2_utils.video_viz import HIGHLIGHT_COLOR, _overlay
from experiments.find_corrected_chains import find_corrected_chains
from experiments.report_assets import padded_window

SCALE = 8   # matches report_assets.py's SCALE, the _sam grid these masks are stored on


def _anchor_frame_index(after_tree: Path, neuron: str, chain_idx: int) -> int:
    """Position of the anchor z within the chain's own sorted z-list, the same
    ordering report_assets.py's build_view uses to number gif frames, so "frame N
    of the gif" and this index agree by construction."""
    cdir = after_tree / neuron / f"chain_{chain_idx:02d}"
    st = json.load(open(cdir / "state.json"))
    anchor_z = st["anchor_catmaid_z"]
    masks = pipeline.chain_masks_in_sam(cdir)
    return sorted(masks.keys()).index(anchor_z)


def extract_anchor_png(assets: Path, frames_out: Path, after_tree: Path,
                       neuron: str, chain_idx: int, side: str) -> Path:
    """The anchor frame of an already-rendered before/after gif, as a standalone
    PNG. side is 'before' or 'after'."""
    gif_path = assets / neuron / f"chain_{chain_idx:02d}_{side}.gif"
    out_path = frames_out / f"{neuron}_chain_{chain_idx:02d}_{side}.png"
    idx = _anchor_frame_index(after_tree, neuron, chain_idx)
    im = Image.open(gif_path)
    idx = min(idx, im.n_frames - 1)   # guard: before/after frame counts could differ
    im.seek(idx)
    im.convert("RGB").save(out_path)
    return out_path


def render_uncorrected_start_frame(before_tree: Path, frames_out: Path,
                                   neuron: str, chain_idx: int) -> Path:
    """A single anchor-frame PNG rendered directly from disk for a chain that was
    never re-rendered as a gif (chain-gif-pair only runs for corrected chains),
    because the human reviewed it and the original pipeline output was already
    correct. Same overlay style (HIGHLIGHT_COLOR/_overlay, the functions to_gif/
    to_mp4 themselves use) and the same proportional-padding window (padded_window)
    as the corrected chains' extracted frames, so both look consistent."""
    cdir = before_tree / neuron / f"chain_{chain_idx:02d}"
    st = json.load(open(cdir / "state.json"))
    anchor_z = st["anchor_catmaid_z"]
    masks = pipeline.chain_masks_in_sam(cdir)
    mask, x0, y0 = masks[anchor_z]
    h, w = mask.shape

    em, full_hw = pipeline.load_frame_sam(anchor_z, scale=SCALE)
    H, W = full_hw
    em_rgb = em if em.ndim == 3 else np.stack([em] * 3, axis=-1)

    wx0, wy0, wx1, wy1 = padded_window((x0, y0, x0 + w, y0 + h), full_hw)
    em_win = em_rgb[wy0:wy1, wx0:wx1]

    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y0:y0 + h, x0:x0 + w] = mask
    mask_win = full_mask[wy0:wy1, wx0:wx1]

    overlaid, _ = _overlay(em_win, mask_win, HIGHLIGHT_COLOR, alpha=0.5)
    out_path = frames_out / f"{neuron}_chain_{chain_idx:02d}_uncorrected.png"
    Image.fromarray(overlaid.astype(np.uint8)).save(out_path)
    return out_path


def _neuron_chain_count(tree: Path, neuron: str) -> int:
    return sum(1 for p in (tree / neuron).glob("chain_*") if p.is_dir())


def build_table(doc: Document, before_tree: Path, after_tree: Path, assets: Path,
                frames_out: Path, neuron: str, corrected: set[int]) -> None:
    n = _neuron_chain_count(after_tree, neuron)
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, text in enumerate(["Chain", "Before (Start Frame)", "After (Start Frame)",
                              "Before (Render)", "After (Render)", "Note"]):
        hdr[i].text = text
    missing_gifs = []
    for c in range(n):
        row = table.add_row().cells
        row[0].text = str(c)
        if c in corrected:
            gif_before = assets / neuron / f"chain_{c:02d}_before.gif"
            gif_after = assets / neuron / f"chain_{c:02d}_after.gif"
            if not (gif_before.exists() and gif_after.exists()):
                # The working tree is live: find_corrected_chains() just found this
                # chain corrected, but report_assets.py chain-gif-pair has not been
                # run for it yet (a real case hit building the first AIAL/AIAR
                # report, a chain got corrected mid-build). Note it and move on
                # rather than crashing the whole in-memory doc, which is only saved
                # once at the very end, losing every row built so far.
                row[5].text = "Corrected, but not yet rendered (run chain-gif-pair for this chain)"
                missing_gifs.append(c)
                print(f"  {neuron} chain_{c:02d}: corrected but gif missing, skipped for now")
                continue
            before_png = extract_anchor_png(assets, frames_out, after_tree, neuron, c, "before")
            after_png = extract_anchor_png(assets, frames_out, after_tree, neuron, c, "after")
            row[1].paragraphs[0].add_run().add_picture(str(before_png), width=Inches(1.3))
            row[2].paragraphs[0].add_run().add_picture(str(after_png), width=Inches(1.3))
            # Render columns: the real content is the animated gif, which cannot
            # survive into docx/Google Docs. A filename pointer, not an image.
            row[3].text = f"{neuron}/chain_{c:02d}_before.gif"
            row[4].text = f"{neuron}/chain_{c:02d}_after.gif"
        else:
            # No correction was made because the original mask was already right.
            # Before column only, there is no "after", nothing changed, so a
            # reviewer can see it was already correct, not just unreviewed.
            png = render_uncorrected_start_frame(before_tree, frames_out, neuron, c)
            row[1].paragraphs[0].add_run().add_picture(str(png), width=Inches(1.3))
            row[5].text = "Already correct, no changes made"
        print(f"  {neuron} chain_{c:02d}: {'corrected' if c in corrected else 'uncorrected'}")
    return missing_gifs


def build_report(before_tree: Path, after_tree: Path, neurons: list[str], assets: Path,
                 out_path: Path) -> None:
    frames_out = assets / "anchor_frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    corrected_rows = find_corrected_chains(before_tree, after_tree, neurons)
    corrected_by_neuron: dict[str, set[int]] = {n: set() for n in neurons}
    for neuron, chain_idx, _anchor_z in corrected_rows:
        corrected_by_neuron[neuron].add(chain_idx)

    doc = Document()
    doc.add_heading("Pipeline Report", level=0)
    note = doc.add_paragraph()
    note.add_run(
        "Start Frame columns show the ANCHOR frame (the frame that was actually "
        "corrected), one static image per chain. Render columns name the animated "
        "gif for that chain instead of embedding it, since neither Word nor Google "
        f"Docs plays animated GIFs; the real files are under {assets}\\<neuron>\\ "
        "if you want to see the actual motion."
    ).italic = True

    all_missing: dict[str, list[int]] = {}
    for neuron in neurons:
        doc.add_heading(neuron, level=1)
        doc.add_paragraph("Correction Time: ")
        doc.add_paragraph("Improvements (According to Metrics)")
        doc.add_paragraph("[Merged Render, before and after]")
        print(f"=== {neuron} ===")
        missing = build_table(doc, before_tree, after_tree, assets, frames_out, neuron,
                              corrected_by_neuron[neuron])
        if missing:
            all_missing[neuron] = missing
        doc.add_page_break()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"wrote {out_path}")
    if all_missing:
        print("[build-docx-report] some corrected chains had no rendered gif yet, noted in "
             "the doc instead of embedding an image, run report_assets.py chain-gif-pair for "
             "these and re-run this script:")
        for neuron, chains in all_missing.items():
            chain_list = ",".join(f"{c:02d}" for c in chains)
            print(f"  {neuron}: {chain_list}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="original/source tree")
    ap.add_argument("--after", required=True, help="corrected working tree")
    ap.add_argument("--neurons", required=True, help="comma-separated neuron names")
    ap.add_argument("--assets", required=True,
                    help="report_assets.py chain-gif-pair output root (has <neuron>/chain_NN_{before,after}.gif)")
    ap.add_argument("--out", required=True, help="output .docx path")
    args = ap.parse_args(argv)

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    build_report(Path(args.before), Path(args.after), neurons, Path(args.assets), Path(args.out))


if __name__ == "__main__":
    main()
