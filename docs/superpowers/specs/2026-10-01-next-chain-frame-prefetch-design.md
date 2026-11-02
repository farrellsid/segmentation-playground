# Prefetch the next chain's frames while reviewing the current one

Status: design, approved 2026-10-01.

## Why

Opening the next chain in the review GUI (`next CHAIN` or the picker) blocks on `open_chain`
re-cropping that chain's frames before the viewer appears. For a tier-2 (`_pcrop`) chain this is
genuinely slow: `prepare_chain_crop_frames` has no cross-chain decode cache (every chain's crop
window is unique), and it unconditionally deletes and rebuilds its view directory from scratch on
every call, even when an identical one was already built moments ago. In `--anchor-only` sessions
this runs on every single chain open, narrowing to the anchor z plus `--context-frames` each time,
but still paying the full rebuild cost instead of reusing anything.

The fix is two pieces: make that rebuild skippable when nothing actually changed, then use the
idle time while a human is correcting the current chain to do the next chain's rebuild in the
background, so by the time `next CHAIN` is pressed the work is already done.

## Part 1: idempotent `prepare_chain_crop_frames`

Add a sidecar file `_prep_meta.json` inside each view directory
(`chain_views/{neuron}_chain{idx:02d}_pcrop_s{crop_scale}/`), written once the view is fully
built, recording what was actually written:

```json
{"z_range": [start_z, end_z], "window": {<cw.to_dict()>}, "n_frames": N,
 "anchor_frame_idx": idx, "anchor_catmaid_z": anchor_catmaid_z,
 "frame_to_z": {"0": z0, "1": z1, ...}}
```

On entry, `prepare_chain_crop_frames` reads this sidecar (if present) and compares it against the
requested `(z_range, cw, anchor_catmaid_z)`. A match, plus every expected frame file
(`00000.jpg` .. `{n_frames-1:05d}.jpg`) actually present, skips the rmtree+rebuild entirely and
returns `(str(view_dir), frame_to_z, anchor_frame_idx, n_frames)` read straight from the sidecar,
so a cache hit touches no frame-store I/O at all, not even to recompute `frame_to_z`.

The sidecar is written **last**, via a temp file renamed into place
(`_prep_meta.json.tmp` -> `_prep_meta.json`), after every frame is written and `cv2.imwrite`
has returned successfully for each. A prepare that crashes or is killed mid-write leaves no
sidecar (or a stale one that fails the match check against the new request), so a half-built
directory is never mistaken for a valid cache; the normal rmtree+rebuild path still fires in that
case, exactly as it does today for any non-matching request.

`prepare_video_frames` (the non-tier-2 path) is unchanged: it already has a dataset-wide decode
cache (`_ensure_cached_frames`, keyed by z + scale), so a second call only has to relink the per-
chain view, which the function's own docstring already calls free.

## Part 2: background prefetch of "the next chain"

In `gui.py`, `open_chain` gains a call to `self._prefetch_next()` at the end, once the current
chain has finished loading.

`_prefetch_next`:

1. Computes which `(neuron, chain_idx)` `next CHAIN` would open right now, by reusing the exact
   list-and-wraparound logic `_step_chain` already uses (`self._mode_chains()`, same index math),
   factored out into a small `_peek_chain(direction)` helper so both call sites share one
   definition of "what's next." No side effects: it does not touch `self.neuron`/`self.chain_idx`
   or open anything.
2. If there is no such chain (empty queue, or the only chain in the list is the one just opened),
   it does nothing.
3. Otherwise it loads that chain's `state.json` (if any) and starts a daemon
   `threading.Thread` that calls `_resolve_chain_frames` for it, with the session's current
   `self.anchor_only` / `self.context_frames`, i.e. the exact same code path `open_chain` itself
   uses. A chain with no `state.json`, or no recorded `anchor_catmaid_z`, is a no-op in
   `_resolve_chain_frames` already (returns `(None, None, None)`), so prefetching a never-run
   chain costs nothing extra.

Re-armed every time a chain finishes loading, so there is at most one chain's worth of
preparation happening ahead of wherever you currently are; it is not a multi-chain lookahead
queue.

### Concurrency

A `threading.Lock` per `(neuron, chain_idx, scale_or_crop_scale)` key, stored in a dict on the
`ReviewGUI` instance (`self._prep_locks`, created on demand), guards the "check sidecar / else
rebuild" section of `_ensure_local_frames` for that key. Both the background prefetch thread and
the real, synchronous call `open_chain` makes take this lock before touching that chain's view
directory.

If you press `next CHAIN` before the prefetch for it has finished, `open_chain`'s call simply
blocks on the same lock the prefetch thread is holding, rather than starting a second, colliding
rebuild of the same directory. Once the prefetch releases the lock, `open_chain` acquires it,
reads the now-matching sidecar, and returns immediately, no wasted duplicate work either way.

A prefetch thread's exceptions are caught at the top level of the thread function and logged
(`print(f"[gui] background prefetch of {neuron} chain_{chain_idx:02d} failed: {e}")`), never
raised into the GUI thread. A prefetch that ends up targeting a chain you never actually open
(you jumped elsewhere via the picker) just leaves a correctly-built, unread view directory behind;
harmless, and reusable if you do visit that chain later in the same session.

### Non-goals

- No new CLI flag. Prefetch is always on once this lands: it has a safe, silent fallback (worst
  case, no speed win that one time) and nothing to opt into.
- No prefetching backward (`prev CHAIN`) and no multi-chain lookahead buffer, depth is always 1.
- No change to `--anchor-only` / `--context-frames` semantics, navigation keybindings, or the
  picker's "flagged" vs "everything" modes.

## Testing

- `pipeline/crop.py`: a CPU-only test that calls `prepare_chain_crop_frames` twice with identical
  arguments against a small fixture tif stack, monkeypatching `cv2.imwrite` to count calls, and
  asserts the second call makes zero `cv2.imwrite` calls (pure cache hit) while still returning
  the same `(frames_dir, frame_to_z, anchor_frame_idx, n_frames)`. A second test changes one
  field (`z_range`, or the crop window) between calls and asserts the second call DOES rebuild
  (cache correctly invalidated, not just always-skip).
- `gui.py`: a test for `_peek_chain(direction)` against the existing `_mode_chains` fixtures
  (same pattern as whatever currently exercises `_step_chain`/`open_next_in_queue`), covering: a
  queue of 1 (returns None), wraparound at both ends, and flagged-vs-everything mode. A test that
  `_prefetch_next` starts a thread targeting the correct chain and that an exception inside it is
  caught and logged rather than propagated (patch the threaded function to raise, assert no
  exception escapes `open_chain`).
