"""Write meta.json into an output tree that predates it.

The corrected AIA/AIY trees were produced before meta.json existed, so they carry
identity only in their directory names. This walks a tree and writes one record
per chain from the chain's own state.json.

    py -3 backfill_meta.py --output-root "F:\\ZhenLab\\Data\\output_masks\\my_tree"
    py -3 backfill_meta.py --output-root ... --backend sam3 --reprop-variant mask_seed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from sam2_utils import chain_meta, registry


def backfill_tree(output_root: Path, *, source_tree: str, backend: str = "sam2",
                  reprop_variant: Optional[str] = None,
                  overwrite: bool = False) -> List[Path]:
    """Write ``meta.json`` for every chain under ``output_root``.

    Parameters
    ----------
    output_root : Path
        A tree laid out as ``<neuron>/chain_<idx>/state.json``.
    source_tree : str
        Provenance label recorded in each record.
    backend : str, optional
        ``"sam2"`` or ``"sam3"``.
    reprop_variant : str, optional
        ``"mask_seed"`` or ``"box_seed"`` for a re-propagation tree.
    overwrite : bool, optional
        Rewrite records that already exist. Default False, so a re-run is cheap.

    Returns
    -------
    list of Path
        The records written this call.

    Raises
    ------
    registry.UnknownNeuronError
        If a directory names a neuron that is not registered. That is a mistake
        worth stopping on, not something to invent an id for.
    """
    output_root = Path(output_root)
    reg = registry.load_registry()
    written: List[Path] = []
    for state_path in sorted(output_root.glob("*/chain_*/state.json")):
        chain_dir = state_path.parent
        if (chain_dir / chain_meta.META_FILENAME).exists() and not overwrite:
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        nid = registry.neuron_id(state["neuron"], registry=reg)
        meta = chain_meta.build_meta(state, neuron_id=nid, source_tree=source_tree,
                                     backend=backend, reprop_variant=reprop_variant,
                                     chain_dir=chain_dir)
        written.append(chain_meta.write_meta(chain_dir, meta))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--source-tree", default=None,
                    help="provenance label; defaults to the output root's folder name")
    ap.add_argument("--backend", default="sam2", choices=["sam2", "sam3"])
    ap.add_argument("--reprop-variant", default=None, choices=["mask_seed", "box_seed"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    written = backfill_tree(args.output_root,
                            source_tree=args.source_tree or args.output_root.name,
                            backend=args.backend, reprop_variant=args.reprop_variant,
                            overwrite=args.overwrite)
    print(f"[backfill] wrote {len(written)} meta.json record(s) under {args.output_root}")


if __name__ == "__main__":
    main()
