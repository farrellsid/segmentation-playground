# Launcher machine setup, and a recrop that survives the round trip

**Date:** 2026-08-27
**Status:** approved, ready for planning

## The problem

A second reviewer works from a review bundle on her own Mac. The bundle was built to
open on a machine that has none of this project's data, and it succeeds at that: masks,
frames, QC, and the CATMAID slice all travel with it. What does not travel is everything
the *model* needs.

Three consequences, which are the three things asked for:

1. **Recrop is unreachable.** `gui.recrop_chain` calls `run_chain`, which re-reads
   full-resolution tifs to cut a new `_pcrop` window. Those come from
   `config.WORM_PATH`, which the launcher never sets and the bundle never carries.
2. **The raw EM path cannot be recorded.** The launcher stores `frames_root` in its
   profile but shows no field for it, and has no field at all for the raw EM. Both are
   env vars (`SAM2_FRAMES_ROOT`, `SAM2_WORM_PATH`) that today only a shell can set.
3. **SAM2 is gated on a bare `import torch`.** Full mode unlocks on that alone, so a
   machine with torch and no checkpoint enables the reprop controls and fails minutes
   later, and a machine with no raw EM enables recrop and fails deep inside a tif read.

The reviewer already has the raw EM locally, so no part of this is about shipping
26.8 GB of tifs. It is about recording where they are, and refusing to pretend when
they are not.

### The hazard this exposes

Recrop rewrites a chain's **geometry**, and geometry is exactly what the bundle round
trip does not carry. `import_bundle` merges `REVIEWER_OWNED = ("masks", "qc.csv")` plus
two row-wise ledgers. `state.json` is deliberately excluded, because it holds a
bundle-relative `frames_dir` that would break the master tree.

So a recrop done in a bundle today produces masks in a new `_pcrop` window, and only the
masks come home. The master tree's `state.json` still describes the old window, and
every consumer places the new pixels at the old offsets, silently. That is the same
failure `_recrop_to_window`'s own comments warn about, one level out.

A second, sharper form of it: `prepare_chain_crop_frames` namespaces its view directory
by neuron, chain and crop scale, so a **new window at the same scale reuses the same
directory name**. After a naive import the master's recorded `frames_dir` still exists
and still holds the old window's frames, so the GUI would draw the new masks on them
with nothing missing to signal the mismatch.

## Goals

- Record the raw EM path, frames cache, and checkpoint directory from the launcher, so a
  machine holding this data needs no shell and no edit to tracked source.
- Tell a reviewer what her machine can and cannot do **before** a session, naming the fix
  for each thing that is missing.
- Let her recrop inside a bundle, see the result, and have that result come home
  correctly.

## Non-goals

- Shipping raw EM with a bundle. She has it.
- Choosing the model size from the launcher. `PipelineConfig.model_size` stays `large`,
  so her masks stay comparable with the batch. The preflight names which checkpoint that
  implies; it does not offer a cheaper one.
- Making SAM2 fast on MPS. The preflight reports the device honestly and says the
  upstream caveat out loud. Nothing here changes inference speed.
- A bundle schema bump. See Decisions.

## Architecture

Six components, each testable on its own. The pure ones carry the logic and the Qt
window stays a thin shell over them, which is the split `launcher.py` already uses to
make `build_launch_kwargs` testable without a display.

### 1. `sam2_utils/config.py`: one env-overridable constant

`CHECKPOINT_DIR` becomes `Path(os.environ.get("SAM2_CHECKPOINT_DIR", "checkpoints"))`,
matching the pattern `WORM_PATH`, `OUTPUT_ROOT` and `FRAMES_ROOT` already use. Today it
is a bare relative `Path("checkpoints")`, so which checkpoint a session finds depends on
the working directory it was launched from.

### 2. `launcher.py`: the machine settings

Three path rows in a group below the bundle picker, stored in the profile beside the
keys it already keeps:

| profile key | env var | what it is |
|---|---|---|
| `worm_path` | `SAM2_WORM_PATH` | raw EM tif stack, read by recrop |
| `frames_root` | `SAM2_FRAMES_ROOT` | where prepared JPEG frames are cached (already stored, never shown) |
| `checkpoint_dir` | `SAM2_CHECKPOINT_DIR` | where SAM2 checkpoints live or land |

`apply_profile_env` exports all three. `DEFAULT_PROFILE` gains the two new keys with
empty defaults, so an older profile file still loads (`load_profile` already fills from
defaults).

### 3. `launcher.machine_checks(profile) -> list[MachineCheck]`

Pure, no Qt, no torch import at module level. `MachineCheck` is a small dataclass:
`name`, `ok` (bool), `detail` (what was found), `fix` (what to do about it, empty when
ok). The window renders the list; tests assert on it directly.

The checks, in order:

1. **torch** present, with its version in `detail`. The fix names the requirements file.
2. **device**, from `setup.setup_device(verbose=False)`: `cuda`, `mps` or `cpu`. Never
   `ok=False`, because CPU still works. `mps` and `cpu` carry a detail saying propagation
   will be slow, and `mps` adds the upstream note that SAM2 on MPS is preliminary and may
   differ from CUDA.
3. **checkpoint** for `PipelineConfig.model_size` in the checkpoint directory. Missing is
   `ok=True` with a detail saying it will download on first use and roughly how big it
   is, because `setup.ensure_checkpoint` downloads rather than failing. A checkpoint
   directory that cannot be created is `ok=False`.
4. **raw EM**: the path exists, is a directory, and holds at least one file matching the
   stack's `*z<digits>*.tif` naming. A directory of unrelated tifs is not a pass.
5. **frames cache**: exists or can be created, and is writable, proved by writing and
   deleting a temporary file rather than by inspecting permission bits.

### 4. Gating, in the launcher and in the GUI

- The mode combo enables full mode on **torch plus a usable checkpoint directory**, not
  on torch alone. Its tooltip names whichever is missing.
- `build_launch_kwargs` raises for full mode without torch, as it does today. Nothing
  else moves into it: it stays the function that turns a profile into `gui.launch`
  arguments.
- `gui.recrop_chain` and `gui.confirm_recrop` check the raw EM up front and print a
  message naming the launcher field, instead of failing inside a tif read.

### 5. Recrop keeps the bundle a bundle

After `_recrop_to_window`'s `run_chain`, when the output root is a bundle
(`bundle.BUNDLE_MANIFEST` exists at its root):

1. Move the newly prepared frames into `<chain_dir>/frames/`, replacing what was there.
   A new helper, `bundle.adopt_chain_frames(chain_dir, frames_dir) -> str`, does this and
   returns the relative name `"frames"`. Move rather than copy: the source is a
   regenerable cache, and she is on a laptop.
2. Set `state.frames_dir` to that relative name before `save_state`, so
   `validate_bundle` still passes and the chain still opens on her machine.
3. Rewrite `meta.json`, whose `mask_space`, `mask_scale`, `crop_window` and `z_range` a
   recrop invalidates.

In an output tree none of this runs, and the behaviour is exactly what it is today.

### 6. `import_bundle`: carry the geometry home

For each chain the bundle holds, compare its `state.json` geometry against the master's.
When it differs, the chain was recropped, and the import:

1. Merges an allowlist into the master's `state.json`: `crop_window`, `frame_to_z`,
   `n_frames`, `anchor_frame_idx`, `prompts`. The file is merged field by field and
   never copied, the same rule the two root ledgers already follow. `prompts` belongs
   here for the same reason `crop_window` does: a tier-2 chain's seed points and box
   are stored in `_pcrop` pixels, the crop window's own space, not `_sam` (see
   `pipeline/orchestrator.py`'s anchor-phase box seeding). Leaving it out merges the
   new window in beside the old window's prompt points, and a re-predict then runs
   from a positive point no longer on the cell.
2. Copies the bundle's `meta.json` over the master's, since it is a pure projection of
   the same chain's new geometry.
3. Clears the master's `frames_dir`, because the stale view directory would otherwise be
   reused (see the hazard above). The next open regenerates from the raw EM, which the
   master machine has.
4. Reports the change in both dry-run and real output, naming the old and new window
   sizes, because a geometry change is a bigger event than a mask edit and should not
   scroll past looking like one.

`gui._ensure_local_frames` is hardened for a `None` frames dir, which today reaches
`resolved / "00000.jpg"` and raises instead of regenerating. "Nothing recorded" becomes a
supported state rather than a crash.

## Data flow

    she recrops in the bundle
      run_chain writes masks in the new _pcrop window
      frames move into <chain_dir>/frames/, frames_dir goes relative
      state.json + meta.json record the new window
      the bundle still validates, and she keeps reviewing that chain
    she returns the bundle
      import merges masks + qc.csv (unchanged behaviour)
      import detects the geometry difference
      master state.json takes crop_window/frame_to_z/n_frames/anchor_frame_idx/prompts
      master meta.json is replaced
      master frames_dir is cleared, so the next open re-preps from raw EM
      config is never merged

## Decisions

**No schema bump.** The on-disk layout does not change, and the AIA, AIY, AIB, AIZ and
AUA bundles already delivered are version 1. Gating the geometry merge on a version would
silently drop a recrop made in a bundle she already holds. Import detects a geometry
difference per chain instead of trusting a version number.

**Merge geometry, never `config`.** `state_to_dict` serialises the whole
`PipelineConfig`, including `output_root` and `frames_root`. Copying that block back
would write her Mac's paths into the master tree. The allowlist is the mechanism that
prevents it, so it is stated as a list and tested as one.

**Frames move, not copy.** A prepared view directory is a cache that
`prepare_chain_crop_frames` rebuilds fresh anyway, and a duplicated tier-2 chain's frames
are real disk on a laptop.

**The preflight does not download.** It reports a missing checkpoint as a pass with a
size warning. Kicking off an 860 MB download from a settings check is a surprise, and
`ensure_checkpoint` already downloads at first genuine use.

## Error handling

- Every check failure carries the fix, phrased as the launcher field to fill or the
  command to run. No check raises: `machine_checks` catches per check so one failure
  cannot hide the rest, the same reason `validate_bundle` returns a list.
- Recrop with no raw EM configured refuses before touching the model, naming the field.
- `adopt_chain_frames` refuses to run against a directory with no prepared frames, so a
  failed prep cannot empty a chain's `frames/` and leave the bundle unopenable.
- Import treats a geometry change with no `crop_window` on either side as a plain mask
  merge, since a `_sam` chain has no window and that is not an error.

## Testing

Every unit here is reachable without Qt, without torch, and without a GPU, which is what
makes the existing launcher and bundle tests fast.

- `machine_checks`: torch absent, checkpoint missing, checkpoint directory unwritable,
  raw EM absent, raw EM present but holding no stack-named tifs, frames cache unwritable,
  and the all-pass case. Device reporting for each of `cuda`, `mps`, `cpu` with
  `setup_device` stubbed.
- Profile: the two new keys round-trip, `apply_profile_env` exports all three vars, and a
  profile written by the older version still loads.
- `adopt_chain_frames`: frames land in `<chain_dir>/frames/`, the old ones are gone, the
  return value is relative, and an empty source refuses.
- Recrop in a bundle, with `run_chain` stubbed: `frames_dir` ends relative, `meta.json`
  is refreshed, and `validate_bundle` passes afterwards. The same path in an output tree
  leaves the absolute `frames_dir` alone.
- Import: the allowlist merges, `config` does not, `frames_dir` is cleared, `meta.json`
  is replaced, an unchanged chain merges exactly as it does today, and dry-run names the
  geometry change.
- Regression: `_ensure_local_frames` with `frames_dir=None` regenerates rather than
  raising.
