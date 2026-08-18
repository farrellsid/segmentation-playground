"""Build data/neuron_registry.csv once, from the distinct cell names in chains.json.

Run this exactly once. After the CSV is committed, ids are frozen: a new neuron
appends a row with the next free id, and nothing is renumbered, because a
renumber silently invalidates every export already produced. The script refuses
to overwrite an existing registry for that reason; use --append to add newly
seen neurons instead.

    py -3 scripts/build_neuron_registry.py
    py -3 scripts/build_neuron_registry.py --append
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2_utils import registry  # noqa: E402


def distinct_cell_names(chains_path: Path) -> list:
    """Return the sorted distinct ``cell_name`` values in a chains.json file."""
    records = json.loads(chains_path.read_text(encoding="utf-8"))
    return sorted({r["cell_name"] for r in records if r.get("cell_name")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "chains.json")
    ap.add_argument("--out", type=Path, default=registry.REGISTRY_PATH)
    ap.add_argument("--append", action="store_true",
                    help="add newly seen neurons to an existing registry, keeping every id")
    args = ap.parse_args()

    names = distinct_cell_names(args.chains)
    today = date.today().isoformat()

    if args.out.exists() and not args.append:
        raise SystemExit(f"{args.out} exists. Ids are frozen; pass --append to add new neurons.")

    existing = registry.load_registry(args.out) if args.out.exists() else {}
    next_id = max(existing.values(), default=0) + 1
    rows = [{"neuron_id": nid, "cell_name": name, "first_seen": today, "notes": ""}
            for name, nid in sorted(existing.items(), key=lambda kv: kv[1])]
    added = 0
    for name in names:
        if name in existing:
            continue
        rows.append({"neuron_id": next_id, "cell_name": name, "first_seen": today, "notes": ""})
        next_id += 1
        added += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=registry.REGISTRY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[registry] wrote {args.out} ({len(rows)} neurons, {added} new)")


if __name__ == "__main__":
    main()
