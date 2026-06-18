"""
pull_worm.py: fetch + process a worm's CATMAID skeletons into the pipeline's
node table (aggregate_data_pv.csv) + chain decomposition (chains.json / roots.json).

The script form of `Copy_of_Get_catmaid_information_lucinda.ipynb`, generalized so
any worm is one command. The original worm (CATMAID project 336) produced
`data/aggregate_data_pv.csv`; the cross-worm GT is project **280**.

    python pull_worm.py --project-id 280 --out-dir data/groundtruth/skeletons_p280
    python pull_worm.py --project-id 280 --start 0 --end 850     # clip to imaged z

Stages (all in sam2_utils.skeletons; see that module's docstring):
    pull_aggregate -> insert_virtual_nodes -> decompose_chains -> write 3 files.

z-range (--start/--end)
-----------------------
Clips chains to the imaged sections (the GT masks only exist there). Default: no
clip: the script prints the pulled z-distribution so you can pick the range, then
re-run with --start/--end. For project 280 the VAST GT stack is slices 0..850.

Coordinate frames
------------------
Node x/y/z come out in **CATMAID stack pixels** (nm ÷ STACK_RESOLUTION_NM). Landing
them in a worm's image/mask grid is a per-worm registration step (an affine; see
sam2_utils.alignment / config.M_AFFINE for the original worm). That transform is NOT
applied here: this script only produces the raw stack-px node table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sam2_utils import config
from sam2_utils.catmaid import Catmaid
from sam2_utils import skeletons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", type=int, required=True,
                    help="CATMAID project id (original worm = 336; cross-worm GT = 280)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output dir for aggregate_data_pv.csv / chains.json / roots.json")
    ap.add_argument("--url", default=config.CATMAID_URL)
    ap.add_argument("--token", default=None, help="CATMAID API token (else config/.env)")
    ap.add_argument("--allowance", type=int, default=50,
                    help="max z-gap (sections) to bridge with virtual nodes")
    ap.add_argument("--start", type=int, default=None, help="clip chains to z >= START")
    ap.add_argument("--end", type=int, default=None, help="clip chains to z <= END")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    catmaid = Catmaid(url=args.url, project_id=args.project_id, api_token=args.token)

    # 1. pull
    agg = skeletons.pull_aggregate(catmaid)
    zmin, zmax = int(agg["z"].min()), int(agg["z"].max())
    print(f"[pull_worm] pulled z-range: [{zmin}, {zmax}]; "
          f"{agg['cell_name'].nunique()} unique cell names")

    # 2. virtual nodes
    agg_pv = skeletons.insert_virtual_nodes(agg, allowance=args.allowance)

    # 3. decompose
    chains, roots = skeletons.decompose_chains(agg_pv, start=args.start, end=args.end)

    # 4. write (same three artifacts as the notebook)
    csv_path = args.out_dir / "aggregate_data_pv.csv"
    agg_pv.to_csv(csv_path, index=False)
    with open(args.out_dir / "chains.json", "w") as f:
        json.dump(chains, f)
    with open(args.out_dir / "roots.json", "w") as f:
        json.dump(roots, f)

    print(f"[pull_worm] wrote:\n"
          f"  {csv_path}  ({len(agg_pv)} nodes)\n"
          f"  {args.out_dir / 'chains.json'}  ({len(chains)} chains)\n"
          f"  {args.out_dir / 'roots.json'}  ({len(roots)} roots)")


if __name__ == "__main__":
    main()
