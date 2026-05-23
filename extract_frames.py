"""
Extract frames from a video file (.mp4 or .avi) into an output folder.

All relative paths are resolved relative to this script's location,
not the current working directory.

Usage:
    python extract_frames.py <video_path> [output_folder] [--every N]

Examples:
    python extract_frames.py video.mp4
    python extract_frames.py video.avi frames/
    python extract_frames.py video.mp4 frames/ --every 5   # save every 5th frame
"""

import argparse
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the script's directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def extract_frames(video_path: str, output_folder: str, every_n: int = 1) -> int:
    video_path = resolve_path(video_path)
    output_folder = resolve_path(output_folder)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video:  {video_path}")
    print(f"Output: {output_folder}")
    print(f"Total frames in video: {total}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n == 0:
            out_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to '{output_folder}'")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video", help="Path to .mp4 or .avi file (relative to script dir if not absolute)")
    parser.add_argument("output", nargs="?", default="frames", help="Output folder (default: ./frames next to script)")
    parser.add_argument("--every", type=int, default=1, help="Save every Nth frame (default: 1)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.every)


if __name__ == "__main__":
    main()
