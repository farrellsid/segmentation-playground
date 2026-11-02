# export_bundle.py --no-frames: skip shipping frames to a raw-EM-capable reviewer

Status: design, approved 2026-10-09.

## Why

`export_bundle.py` always regenerates and embeds every chain's frame JPEGs, even though
frames are ~200x the size of everything else in a bundle and, per `review-over-git.md`,
travel by drive once rather than through git. Measured on real data: exporting the `AIA`
pair (2 neurons, 66 chains) took 24m23s, almost entirely in frame regeneration from the
raw EM (`caching _pcrop frames`, ~3s/frame).

A reviewer who already has her own raw EM copy and has gone through
`set-up-sam2-on-a-review-machine.md` (torch, checkpoint, raw EM, for recrop) does not need
those frames shipped at all. `gui.py`'s `_ensure_local_frames` already has a generic
fallback, not recrop-specific: whenever a chain's recorded `frames_dir` is falsy (`None`
or missing), it regenerates the frames locally from the raw EM via
`pipeline.prepare_chain_crop_frames`/`prepare_video_frames`, the exact same mechanism
`export_bundle.py` uses on the exporting side. `import_bundle.py` already relies on this:
it sets `frames_dir` to `None` on every chain a recrop brings home, specifically so the
stale view gets regenerated rather than silently reused.

This spec adds a way to deliberately produce that same `frames_dir: None` state at export
time, so a reviewer with her own raw EM gets a bundle in seconds instead of tens of
minutes per neuron, and her own `gui.py` regenerates each chain's frames locally the first
time she opens it.

## The flag

A new `--no-frames` flag on `export_bundle.py`'s CLI and a new `no_frames: bool = False`
keyword on `export_bundle()` itself. Default is unchanged (frames are shipped), since most
reviewers (`review-on-a-mac.md`'s plain install) do NOT have raw EM access and a bundle
without frames is unopenable for them.

**This is a trust-the-caller flag, the same as `--force`/`--force-frames` already are in
this file.** `export_bundle.py` runs on the exporting machine and has no way to verify
anything about the destination reviewer's setup. Using `--no-frames` for someone without
raw EM ships a bundle that silently fails to open any chain. The flag's help text and the
function's docstring say this plainly.

## What changes in `export_bundle()`

In the per-chain loop (currently: check `_frames_complete`, else `regenerate_frames` +
`_replace_frames_dir`, then always `bundle.rewrite_state_frames_dir(rec["state"])`):

- When `no_frames` is set, skip the entire frame block unconditionally: no
  `_frames_complete` check, no `regenerate_frames` call, no `out_dir / "frames"`
  directory ever created. (`force_frames` and `no_frames` together is nonsensical —
  "regenerate frames every time" and "never generate frames" contradict each other;
  `no_frames` wins, since it is the more specific, deliberate ask, and main() validates
  this combination is never passed rather than silently picking a winner.)
- `rewrite_state_frames_dir` already takes an optional `relative` parameter
  (`relative: str = "frames"`) that it assigns directly to `frames_dir` with no other
  logic. Call it as `bundle.rewrite_state_frames_dir(rec["state"], relative=None)` when
  `no_frames` is set, instead of the default call. No change to `rewrite_state_frames_dir`
  itself is needed, it already does exactly this when told to.
- The bundle-wide `bundle.validate_bundle(dest)` call becomes
  `bundle.validate_bundle(dest, require_frames=not no_frames)`. `require_frames=False`
  already exists for exactly this "frames absent and harmless" case (currently used for
  git clones and by `import_bundle.py`); this is the first caller to pass it from
  `export_bundle.py` itself.

Nothing else in `export_bundle()` changes: `_check_source_data_present()` checks the
CATMAID tables (`chains.json`/`nodes.csv`), unrelated to frames; `write_bundle_data` is
the same unrelated data slice; masks/`qc.csv` copying is unaffected.

## CLI

```
py -3 export_bundle.py --output-root <tree> --dest <dir> --neurons AIAL AIAR --no-frames
```

Help text: `"skip shipping frames; only for a destination reviewer who already has her "
"own raw EM and the recrop setup from set-up-sam2-on-a-review-machine.md (gui.py "
"regenerates frames locally on open). Do not use this for a reviewer without raw EM, "
"the bundle will not open."` `main()` raises a clear `SystemExit` if both `--no-frames`
and `--force-frames` are passed together.

## Docs

Add a short section to `review-over-git.md`, after "Building the repository", for a
raw-EM-capable reviewer: `export_bundle.py --no-frames` straight into the git repo
clone's structure (no separate "copy without frames" step, since there are no frames to
strip), commit, push. No `place_frames.py` step on her side either, since there is
nothing to place; her first `gui.py` open of each chain regenerates it from her own raw
EM. The existing frames-shipped path stays documented unchanged for reviewers without
raw EM.

## Non-goals

- No automatic detection of whether a destination reviewer has raw EM. This is a
  deliberate, documented, human decision each time, not something `export_bundle.py` can
  verify from the exporting machine.
- No change to `gui.py`/`_ensure_local_frames`/`import_bundle.py`. The fallback this
  relies on already exists and already works; this spec only adds a second, deliberate
  way to produce the `frames_dir: None` state that triggers it.
- No change to the plain, no-raw-EM review path (`review-on-a-mac.md`,
  `review-over-git.md`'s existing frames-shipped section). Both keep working exactly as
  documented today; `--no-frames` is purely additive.
