# Export a review bundle for Lucinda

For the sender's side: building a bundle with `export_bundle.py` and getting it to her. For
what she does with it, see [review-on-a-mac.md](review-on-a-mac.md). For where bundles live
and how they travel, see `F:\Lucinda_Review\START_HERE.md` (not in git, drive only).

## Picking `--output-root`

Bundle the whole neuron, not just whatever changed, so she is reviewing a complete unit:

```bash
py -3 export_bundle.py --output-root <base tree with every chain> \
    --overlay <tree with only the corrected/re-propagated chains> \
    --neurons AIML --dest F:\Lucinda_Review\bundles\AIM \
    --backend sam3 --reprop-variant mask_seed
```

`--overlay` is for exactly this shape of run: a re-propagation writes only the chains whose
seed was corrected, so give it the tree holding every chain as `--output-root` and the
re-propagation tree as `--overlay`. Later overlays win, and each chain's `meta.json` records
which tree it actually came from, not just the bundle-wide default.

The base tree for most neurons is
`F:\ZhenLab\Data\output_masks\resolution_experiments\target_perslice_only_guard_sam3_merged`,
the ADR 0017 whole-set winner.

**RIH is the exception.** Its own `PROVENANCE.txt` names
`target_tier2_s1forced_neg_sam3_merged` as the source, but that merged tree does not have RIH
locally (checked directly: no `RIH/` directory under it). Use `manual_verify_RIH` as
`--output-root` instead, still with the reprop tree as `--overlay`. Check
`PROVENANCE.txt` inside any `manual_verify_*` tree before assuming the standard base tree
applies; RMDDL had the same substitution for the same reason.

If a base tree turns out to be missing a neuron entirely, `bundle.index_chains` fails loudly
("no chains found") rather than silently writing an empty bundle. That is the intended
behaviour, not a bug to route around: find the real source tree instead.

## Verify before trusting the log

`export_bundle.py` prints `wrote N chain(s)` on success, but that line does not prove the
files are actually on disk. A drive that disconnects between the write and the OS flushing
it to physical media (a laptop sleeping mid-export, a USB drop) can leave the log looking
clean while the bundle is missing most of its files, `bundle.json` included. This happened
for real on 2026-09-09: a log claiming `wrote 41 chain(s)` for an 8-file bundle. Always
check, after every export, not just the ones that looked like they failed:

```bash
py -3 -c "
from pathlib import Path
from sam2_utils import bundle
p = Path('F:/Lucinda_Review/bundles/AIM')
print('validate_bundle:', bundle.validate_bundle(p) or 'clean')
print('state.json count:', len(list(p.glob('*/chain_*/state.json'))))
"
```

Compare the count against what you asked for (`len(bundle.index_chains(output_root, neurons=[...]))`
before exporting, if you want the exact expected number). A mismatch means re-run with
`--force` on that specific bundle, not a rebuild of everything.

## Do not run an export across a machine sleep

Frame regeneration for a full neuron takes real wall-clock time (tens of chains, seconds to
tens of seconds each). If the machine sleeps mid-run, the external drive drops, and whatever
was mid-write is at risk exactly as described above. Keep the machine awake for the
duration, or split the run into smaller per-neuron exports you can checkpoint and verify
between.

Running two exports at once is fine (used successfully for FLP and RIB in parallel).
Going much beyond that is not recommended: `F:` is a known-flaky drive under concurrent
load, a separate real failure mode from the sleep issue above (see the
[2026-08-14 changelog entry](../CHANGELOG.md#r-2026-08-14-reprop-comparison) for the
earlier incident this was first found in).

## Sending it

Zip the bundle folder (plain name, e.g. `AIM.zip`, not `AIM_for_lucinda.zip`, the old
git-based naming) and upload it to the Drive folder linked from
`F:\Lucinda_Review\START_HERE.md`. Tell her which neurons landed and that they are new
(first bundle, not a re-review), so she knows to expect a full pass rather than spot-checks.
