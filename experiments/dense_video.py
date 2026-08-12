"""Scroll-through video of the dense per-frame labelmap across the whole worked volume.

Same artifact as the current-work slide (every SAM3 per-slice neuron mask merged into one
dense labelmap on a frame), but rendered for every z in the covered band and stacked into a
video that scrolls down the stack. LOCAL only: no model, no GPU, no Narval. It reuses the
cached dense-overlay index and the exact merge + colourise path from experiments/dense_overlay,
so a neuron keeps one colour for the whole scroll.

Two phases, split so the slow part never has to repeat:

  render:  read masks from disk, build each frame's overlay PNG into a numbered sequence.
  encode:  stitch the PNG sequence into mp4 (cv2 mp4v) and/or an animated GIF (PIL).

    py -3 experiments/dense_video.py                       # render all z, then encode gif+mp4
    py -3 experiments/dense_video.py --phase render        # just the PNG sequence
    py -3 experiments/dense_video.py --phase encode         # re-encode from existing PNGs

cv2's bundled ffmpeg here can only encode mp4v (openh264 is broken), and mp4v does not play
reliably in Chromium, so GIF is the default slide-safe output. Pass --backend mp4v for the
small archival mp4 as well.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

import pipeline
from experiments import dense_overlay as do

DEFAULT_FRAMES = Path("docs/figures/presentation/dense-scroll/frames")
DEFAULT_OUT = Path("presentation/public/videos")


def render_frames(idx: dict, zs: list[int], scale: int, frames_dir: Path,
                  max_width: int, label: bool) -> tuple[list[int], tuple[int, int]]:
    """Write one overlay PNG per z. Every frame is placed on a fixed canvas (the first
    frame's scale-`scale` size) so the sequence has constant dimensions; frames share the
    scale-`scale` origin, so top-left placement keeps the stack registered. Returns the
    z list actually rendered and the (H, W) canvas used before any max-width downscale."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    lut = do.build_palette(len(idx["neuron_id"]))
    recs = idx["records"]
    z_neurons = {int(zk): {recs[ri]["neuron"] for ri in ris}
                 for zk, ris in idx["z_to_recs"].items()}

    canvas_hw: tuple[int, int] | None = None
    rendered: list[int] = []
    t0 = time.time()
    for i, z in enumerate(zs):
        if z not in z_neurons:
            print(f"[video] z={z}: no masks; skip")
            continue
        em, _full_hw = pipeline.load_frame_sam(z, scale=scale)
        h8, w8 = em.shape[:2]
        if canvas_hw is None:
            canvas_hw = (h8, w8)
        H, W = canvas_hw
        nmasks = do.neuron_masks_at_z(z, idx, scale, h8, w8)
        if not nmasks:
            print(f"[video] z={z}: empty after remap; skip")
            continue
        label_map, _count, _order = do.stack_labelmap(nmasks, idx["neuron_id"], h8, w8)
        over = do.colorize_over_em(em, label_map, lut, alpha=0.5)  # HxWx3 uint8 RGB

        # place on the constant canvas (crop if bigger, pad with black if smaller)
        canvas = np.zeros((H, W, 3), np.uint8)
        hh, ww = min(H, over.shape[0]), min(W, over.shape[1])
        canvas[:hh, :ww] = over[:hh, :ww]

        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        if max_width and W > max_width:
            new_w = max_width
            new_h = int(round(H * max_width / W))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if label:
            txt = f"z = {z}   ({len(nmasks)} neurons)"
            cv2.putText(bgr, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frames_dir / f"f_{len(rendered):04d}.png"), bgr)
        rendered.append(z)
        if (i + 1) % 20 == 0 or i == len(zs) - 1:
            el = time.time() - t0
            print(f"[video] {len(rendered)} frames ({el:.0f}s, {el / max(1, len(rendered)):.2f}s/frame)")
    return rendered, canvas_hw or (0, 0)


def encode_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int, crf: int = 24) -> None:
    """Browser-friendly H.264 mp4 via the static ffmpeg that ships with imageio-ffmpeg.
    yuv420p needs even dimensions, so pad odd sizes up by one pixel; +faststart moves the
    moov atom to the front so it starts playing before the whole file is buffered. Higher
    crf = smaller file; EM texture is noise-heavy so crf 24-28 cuts size a lot with little
    visible loss (the still image slide is the crisp reference)."""
    import subprocess
    import imageio_ffmpeg
    files = sorted(frames_dir.glob("f_*.png"))
    if not files:
        print("[video] no frames to encode"); return
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart",
           str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[video] ffmpeg FAILED:\n" + r.stderr[-1500:]); return
    print(f"[video] wrote {out_mp4} ({len(files)} frames, {fps} fps, "
          f"{out_mp4.stat().st_size / 1e6:.1f} MB)  [H.264, plays in Chromium]")


def encode_mp4(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    files = sorted(frames_dir.glob("f_*.png"))
    if not files:
        print("[video] no frames to encode"); return
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        vw.write(cv2.imread(str(f)))
    vw.release()
    print(f"[video] wrote {out_mp4} ({len(files)} frames, {w}x{h}, {fps} fps, "
          f"{out_mp4.stat().st_size / 1e6:.1f} MB)  [mp4v: may not play in Chromium]")


def encode_gif(frames_dir: Path, out_gif: Path, fps: int, gif_width: int) -> None:
    from PIL import Image
    files = sorted(frames_dir.glob("f_*.png"))
    if not files:
        print("[video] no frames to encode"); return
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if gif_width and im.width > gif_width:
            im = im.resize((gif_width, int(round(im.height * gif_width / im.width))))
        # adaptive 256-colour palette with dithering; keeps the neuron colours legible
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print(f"[video] wrote {out_gif} ({len(files)} frames, {imgs[0].width}px wide, "
          f"{out_gif.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=str(do.DEFAULT_TREE))
    ap.add_argument("--z-start", type=int, default=1293)
    ap.add_argument("--z-end", type=int, default=1628)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=int, default=do.RENDER_SCALE)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--crf", type=int, default=24, help="H.264 quality (higher = smaller)")
    ap.add_argument("--max-width", type=int, default=900,
                    help="cap the rendered PNG width (0 = native)")
    ap.add_argument("--gif-width", type=int, default=760,
                    help="downscale width for the GIF only")
    ap.add_argument("--no-label", action="store_true", help="omit the z counter overlay")
    ap.add_argument("--frames-dir", default=str(DEFAULT_FRAMES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--name", default="dense-scroll")
    ap.add_argument("--phase", choices=["all", "render", "encode"], default="all")
    ap.add_argument("--backend", choices=["ffmpeg", "gif", "mp4v", "both"], default="ffmpeg")
    ap.add_argument("--cache", default="docs/figures/sam3-bakeoff/dense-overlay/_index.json")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args(argv)

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)

    if args.phase in ("all", "render"):
        idx = do.build_index(Path(args.tree), Path(args.cache), rebuild=args.rebuild_index)
        print(f"[video] index: {len(idx['records'])} chains, {len(idx['neuron_id'])} neurons")
        zs = list(range(args.z_start, args.z_end + 1, args.stride))
        rendered, hw = render_frames(idx, zs, args.scale, frames_dir,
                                     args.max_width, label=not args.no_label)
        print(f"[video] rendered {len(rendered)} frames onto {hw[1]}x{hw[0]} canvas "
              f"-> {frames_dir}")

    if args.phase in ("all", "encode"):
        if args.backend in ("ffmpeg", "both"):
            encode_ffmpeg(frames_dir, out_dir / f"{args.name}.mp4", args.fps, args.crf)
        if args.backend in ("gif", "both"):
            encode_gif(frames_dir, out_dir / f"{args.name}.gif", args.fps, args.gif_width)
        if args.backend == "mp4v":
            encode_mp4(frames_dir, out_dir / f"{args.name}.mp4", args.fps)


if __name__ == "__main__":
    main()
