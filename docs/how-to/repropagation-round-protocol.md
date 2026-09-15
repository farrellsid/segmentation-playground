# Repropagation round protocol

Three fixed steps for running a re-propagation round and sending the results to Lucinda.
Follow them in order, every time. Skipping step 1 in favor of a tracking sheet shipped a
batch missing 123 real corrected chains on 2026-09-15; skipping step 2 shipped a bundle
that was 8 of 41 files while its own log claimed success on 2026-09-09. Both are documented
in full in the [CHANGELOG](../CHANGELOG.md).

## 1. Find candidates from metadata, never from a tracking sheet

A neuron's "Initial Corrected" column in a spreadsheet like `SegmentationNotes.csv` is a
human's note about their own work, not a record of what is actually on disk. It can be
wrong. Check the real thing instead: a pixel diff of each chain's anchor mask against the
source tree it started from.

```bash
py -3 experiments/find_corrected_chains.py \
    --source <the neuron's correct base tree, see export-a-lucinda-bundle.md> \
    --working F:\ZhenLab\Data\output_masks\manual_verify_<GROUP> \
    --neurons <NEURONL>,<NEURONR> \
    --out-csv cluster/corrected_chains_<label>.csv
```

Run this for **every side of every neuron pair under consideration**, not just the sides a
sheet claims are ready. On 2026-09-15, `SegmentationNotes.csv` marked AIMR, RIBR, RMHR, and
AVHL as not yet corrected. All four had real corrections on disk (39, 38, 38, and 8 chains),
excluded from the prior round purely because the sheet was trusted over the metadata. The
sheet is a convenience for humans to coordinate; the pixel diff is the source of truth.

## 2. Verify the reprop output is complete and unbroken

Do not treat a Narval array job as done because `sacct` shows every task COMPLETED. Check
the actual output:

- Grep the job logs for the `wrote N masks` line per chain and glance at the distribution.
  A cluster of `1`s across many different chains means something is silently short-circuiting
  (a real incident: a genuinely single-node chain wrote 1 mask correctly, but the same
  pattern across many chains would mean the propagation range is wrong, not the chain).
- After pulling the tarball locally, extract it and check, per neuron, that the chain
  directory count matches the manifest exactly:

  ```bash
  find <extracted_tree>/<NEURON> -maxdepth 1 -type d -name "chain_*" | wc -l
  ```

- Confirm every chain has both a `state.json` and at least one mask PNG, and that no file is
  zero bytes:

  ```bash
  find <extracted_tree> -name state.json | wc -l
  find <extracted_tree> -type f -size 0 | wc -l
  ```

- Spot-check that a few masks actually contain painted content, not blank canvases (`cv2.imread`
  a sample, confirm `(img > 0).sum()` is nonzero and a sane fraction of the frame).

This mirrors, at the reprop-output stage, the same discipline
[export-a-lucinda-bundle.md](export-a-lucinda-bundle.md) requires at the bundle-export
stage: a process reporting success is a claim, not a proof.

## 3. Bundles are always full neuron pairs, never one side

Before zipping anything for Lucinda, confirm both the L and R chain directories exist and
are non-empty in the export destination:

```bash
py -3 -c "
from pathlib import Path
p = Path('F:/Lucinda_Review/bundles/<GROUP>')
for d in sorted(p.iterdir()):
    if d.is_dir() and not d.name.startswith('_'):
        n = len(list(d.glob('chain_*')))
        print(d.name, n, 'chains')
"
```

If a genuine reason exists for a lopsided pair (one side truly has zero chains at all,
corrected or otherwise, in the base tree), stop and ask before shipping it. Do not ship a
bundle with only one side of a pair on the assumption that "the other side just was not
ready this round." Step 1, done properly, is what prevents this from being a silent
assumption in the first place.
