"""make_chunks.py: split the target-worm neuron list into per-task chunks.

Writes ``neuron_chunks.txt``, one line per Slurm array task, each line a
space-separated set of neuron names. The array script reads its own line with
``sed -n "$((SLURM_ARRAY_TASK_ID+1))p"`` and passes it to ``batch.py --neurons``.

Chunking by neuron (not by chain) so one SAM2 model load is amortised over a whole
chunk's chains, and so the on-disk ``<neuron>/`` output dirs stay whole within a
single task (no two tasks writing the same neuron).

Usage
-----
    py -3 cluster/make_chunks.py                 # default chunk size 7
    py -3 cluster/make_chunks.py --chunk-size 5 --out cluster/neuron_chunks.txt
    py -3 cluster/make_chunks.py --neurons AVAL AVAR RIS   # explicit subset instead of ALL_NEURONS

The array size to request is the number of lines written (printed at the end):
``#SBATCH --array=0-<N-1>%<concurrency>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

# Run standalone (python cluster/make_chunks.py) or as a module: put the repo root
# on the path so `sam2_utils` resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2_utils import presets


def chunk_neurons(neurons: Sequence[str], chunk_size: int) -> List[List[str]]:
    """Split ``neurons`` into consecutive chunks of at most ``chunk_size``.

    Order is preserved and every neuron lands in exactly one chunk, so the chunks
    are disjoint and cover the whole list (the last chunk may be shorter).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [list(neurons[i:i + chunk_size]) for i in range(0, len(neurons), chunk_size)]


def write_chunks(chunks: Sequence[Sequence[str]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(chunk) for chunk in chunks]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=7,
                    help="neurons per array task (default 7)")
    ap.add_argument("--neurons", nargs="*", default=None,
                    help="explicit neuron list (default: presets.ALL_NEURONS)")
    ap.add_argument("--out", type=Path, default=Path("cluster/neuron_chunks.txt"),
                    help="output path for the chunk file")
    args = ap.parse_args()

    neurons = args.neurons if args.neurons else presets.ALL_NEURONS
    chunks = chunk_neurons(neurons, args.chunk_size)
    write_chunks(chunks, args.out)

    print(f"[make_chunks] {len(neurons)} neurons -> {len(chunks)} chunks "
          f"(<= {args.chunk_size} each) -> {args.out}")
    print(f"[make_chunks] request:  #SBATCH --array=0-{len(chunks) - 1}%<concurrency>")


if __name__ == "__main__":
    main()
