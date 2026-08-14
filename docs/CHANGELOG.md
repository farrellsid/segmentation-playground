# Changelog and history archive

The append-only build log: the milestone-by-milestone narrative, the resolution stories for closed
issues, the full design-decision log (including what was rejected and why), the A/B results, and the
original field notes from first GUI use. This is version-scoped history, so it still uses the
project's old milestone and section vocabulary on purpose.

For the current architecture see [explanation/architecture.md](explanation/architecture.md). For the
load-bearing decisions distilled into one-page records, see [the ADRs](adr/README.md). For the
detailed design rationale and the active backlog, see
[explanation/design-notes.md](explanation/design-notes.md).

**Why it's split out.** design-notes.md is the lean, current-state reference a reader
(or model) loads to understand the design and the live backlog. This archive is the deep
record, read it when you need the *why* behind a past decision, source material for a
report/paper, or the exact numbers from an A/B. Nothing here was deleted; it was moved
here verbatim from design-notes.md (June 2026 reorg).

**Section anchors are preserved.** The original §-numbers are kept (old §2, §5, §7, §8, §9)
so existing cross-references from code comments, the README, and other notes still resolve, they now point here for the historical material. The live design-notes.md keeps §1, §3,
§4, §6 and a trimmed §5 (the gotchas catalogue), and adds a reorganized backlog.

---

## Contents
- [2026-08-14, re-propagation comparison report: mask-seed vs box-seed, a real underfill fix found, transient F:/RAM read failures hardened](#r-2026-08-14-reprop-comparison)
- [2026-08-12, review-session follow-ups: a shared before/after crop window, --anchor-only context frames](#r-2026-08-12-review-followups)
- [2026-08-12, lasso-add mask tool: a freehand loop unions into the current frame's mask, undo-compatible with the paint brush](#r-2026-08-12-lasso-add)
- [2026-08-12, `--anchor-only` prompt-seeding bug: prompts/box placed off the narrowed frame, explaining a "0/56" report](#r-2026-08-12-anchor-only-seed-bug)
- [2026-08-10, propagation from manually verified multi-frame seeds: mechanism verified, driver built](#r-2026-08-10-propagate-from-verified)
- [2026-08-09, spill/overfill documentation prep: real membrane-pass cost measured, a stale CSV trap found, micro_sam and square-mask spot-checks](#r-2026-08-09-spill-doc-prep)
- [2026-08-06, propagation second pass: full-scope validation, a real verdict](#r-2026-08-06-second-pass-fullscope)
- [2026-08-04, NucleoNet recall gap found, MitoNet mitochondria detector spot-checked](#r-2026-08-04-mitonet)
- [2026-07-30, propagation second pass: neighbour-seeded re-segmentation lands, SAM3 crash found and fixed, accept-gate gap flagged](#r-2026-07-30-second-pass)
- [2026-07-29, organelle blob suppression: the intensity/texture filter lands, calibrated against real EM, the floor still does not move](#r-2026-07-29-organelle-blob)
- [2026-07-29, z-to-z consistency metric: a temporal ruler for the merge metric's per-frame blind spot](#r-2026-07-29-z2z)
- [2026-07-29, nucleus-capture detector: GUI label added, NucleoNet spot-checked and verdicted](#r-2026-07-29-nucleus)
- [2026-07-29, temporal membrane projection: register/project crops land, a gate confound found and corrected](#r-2026-07-29-temporal)
- [2026-07-29, SAM3-vs-SAM2 scorecard consolidated, default backend decided](#r-2026-07-29)
- [2026-07-23, sharded parallel scoring, the SAM3 config A/B presets, and a documentation sweep](#r-2026-07-23)
- [2026-07-21, SAM3 Phase 2: `--backend sam3` switch, cluster wiring, and the Narval runbook](#r-2026-07-21-sam3-cluster)
- [2026-07-21, SAM3 vs SAM2 bake-off: HF-transformers PVS adapters + 2x2 comparison](#r-2026-07-21-sam3)
- [2026-07-21, per-frame segmentation: two approaches, an AMG tuner, membrane-aware arbitration](#r-2026-07-21)
- [2026-07-20, Phase 1 close-out: blow-up guard + generous-first/negatives-in-crop bundle](#r-2026-07-20)
- [2026-07-17, Phase 2 foundation: membrane map + membrane-aware bleed detection](#r-2026-07-17)
- [2026-07-15, negatives round + measurement-first roadmap redesign](#r-2026-07-15)
- [2026-07, research passes + resolution experiments + specs](#r-2026-07)
- [2026-06, review tooling + post-processing pass](#r-2026-06)
- [old §2, Milestone-by-milestone build narrative (M1 → M4 review-testing pass)](#old-2)
- [old §5, Known issues: full resolution stories](#old-5)
- [old §7, Design decisions: full log (landed + rejected, with rationale)](#old-7)
- [old §8, M4.5 A/B results & decisions log](#old-8)
- [old §9, Raw field notes from first GUI use (pre-reorg, verbatim)](#old-9)

---

<a id="r-2026-08-14-reprop-comparison"></a>
## 2026-08-14, re-propagation comparison report: mask-seed vs box-seed, a real underfill fix found, transient F:/RAM read failures hardened

Report tooling for the AIA/AIY re-propagation job (2026-08-13, `cluster/run_reprop_corrected_seed.sh`
run on 45 AIA + 42 AIY corrected chains, mask-seed and box-seed variants each): compares the two
seeding strategies against each other and against the pre-reprop tail.

**`experiments/report_assets.py` gained `render_reprop_triple`/`chain-gif-triple` and
`neuron-gif-triple`.** Renders before/mask-seed/box-seed to ONE shared crop window per chain (or,
for `neuron-gif-triple`, per neuron, every reprop'd chain overlaid in one crop), computed from the
union of both reprop trees' masks since neither variant is more "correct" a priori. Also gained
`metrics_reprop`/`metrics-reprop`, a merge-metric comparison restricted to a manifest's reprop'd
chains (not a neuron's full chain list, which would average in dozens of untouched chains and make
the comparison meaningless). `experiments/render_reprop_report.py` (new) batches the per-chain
triples from a manifest CSV, idempotent (skips a chain whose three gifs already exist).
`experiments/build_reprop_docx_report.py` (new) assembles the actual `.docx`: a per-chain table
keyed to each variant's own FARTHEST-from-anchor frame (not the anchor frame, identical across all
three trees by construction, so it would show no difference at all), plus a whole-neuron "Merged
Render" section (middle-frame snapshot, since a multi-chain merge has no single anchor or far end to
key off).

**Real finding: both reprop variants score WORSE than the pre-reprop tail on the Phase-0 bleed
metrics, consistently across all 4 sides (AIAL/AIAR/AIYL/AIYR), but this reads backwards without the
membrane pass.** `foreign_frame_rate`/`total_foreign_nodes` go up in every case (e.g. AIYR
mask-seed: 0.150 -> 0.339). Enabling `membrane_source="auto"` (off by default in the earlier
`metrics_before_after`, on here) surfaces the real story: `mean_underfill_fraction` drops 6-8x in
every case (AIAL: 2.10 -> 0.26; AIYR: 1.69 -> 0.25), and `mean_boundary_on_membrane` improves too
(+0.09 to +0.12). The pre-reprop tail was badly underfilled (visually confirmed: AIYL chain_03's
farthest frame was a thin sliver before, a properly cell-sized mask after, both variants), and reprop
fixes that at the cost of the mask now genuinely reaching neighbouring territory it previously missed
by being wrong in a different way. **Mask-seed beats box-seed on overfill in all 4 sides** while
matching it on the underfill fix, so it remains the better default of the two.

**`pipeline/frames.py::load_frame_sam` gained a bounded retry (8 attempts, 3s apart) around
`cv2.imread`.** Confirmed under real load from this report's whole-neuron merged renders (176-364
full-res ~9216x9230 frames read per run): two distinct failure modes hit the same call site, transient
F: read failures (`imread` returns `None`, the identical file reads fine moments later) and system
RAM exhaustion under concurrent heavy jobs (`imread` raises an explicit `OutOfMemoryError`, one frame
is ~255MB; this machine has 19.7GB total). The retry absorbs both by construction (rides out a disk
hiccup, or gives a concurrent process time to finish and free memory), but does not fix sustained RAM
pressure by itself: avoid running more than one heavy render/score job (including gui.py) at a time.

**Delivery gap found and fixed**: the docx's "Renders" column note pointed at a local F: path
("the real files are under F:\...\<neuron>\"), useless to a supervisor who does not have that drive.
Fixed in both `build_reprop_docx_report.py` and the earlier `build_docx_report.py` to say the gif
files travel alongside the document instead. Delivered as `AIA_reprop_report.zip` /
`AIY_reprop_report.zip` (docx + every referenced gif at the right relative paths, verified by
listing the zip's own root entries against the note text). One correction worth recording: Google
Docs CAN autoplay an animated GIF, but only via "Insert Image > By URL" (a live reference), never for
an uploaded file or a docx import, which always flattens to a static raster; Word never animates a
GIF regardless of insertion method.

343 passed, 1 skipped; ruff clean; no dashes.

---

<a id="r-2026-08-12-review-followups"></a>
## 2026-08-12, review-session follow-ups: a shared before/after crop window, --anchor-only context frames

Two small requests from a real AIAL/AIAR review pass, both landed the same day as the lasso tool.

**`experiments/report_assets.py`'s before/after renders now share one crop window, computed from the
AFTER (revised) tree.** Previously `chain-gif`/`neuron-gif` each computed their own window from
whichever tree they were rendering: the union of that tree's own mask extent across every z, padded.
Called once for a before tree and once for an after tree, as the module's own usage examples showed,
the two windows could drift apart the moment a correction changed the mask's footprint (the lasso tool
extending it outward, or a box-seed re-predict pulling it in), cropping the before render short of
where the after mask now reaches, or vice versa. `_load_chains_masks`/`_compute_window` are pulled out
of `build_view` as their own functions; `build_view`/`render` take an optional `window` (wx0, wy0, wx1,
wy1 in _sam px) and use it as-is instead of recomputing one; a new `render_before_after` computes the
window once from the AFTER tree and renders both sides to it. Two new CLI subcommands,
`chain-gif-pair` and `neuron-gif-pair`, wrap this for the common case (the old single-tree
`chain-gif`/`neuron-gif` are unchanged, still compute their own window, for anyone rendering just one
side). Verified against real data (AIAL chain_00, before tree
`resolution_experiments/target_perslice_only_guard_sam3_merged`, after tree
`manual_verify_AIAL_AIAR`): both renders come back at the identical `378x375`, 113 frames each, not a
coincidence of the code, the shared window forces it.

**`gui.py --anchor-only` gained `--context-frames <n>`.** A real gap surfaced while reviewing: seeing
only the anchor frame in isolation makes it hard to judge whether a correction actually matches how
the mask should look a slice or two either way. `--context-frames n` widens the prepared z_range from
`(anchor_z, anchor_z)` to `(anchor_z - n, anchor_z + n)`, so `,`/`.` can scrub a little context around
the seed without paying for the whole chain's regeneration cost `--anchor-only` exists to avoid.
Threaded through `_ensure_local_frames` (only takes effect when `anchor_only` is also set, ignored
otherwise) down to `ReviewGUI.__init__`, `launch()`, and a new `main()` CLI flag. No new bug this
time: `prepare_chain_crop_frames`/`prepare_video_frames` already compute the correct LOCAL anchor
index for whatever `z_range` they are given (this is the same code path `anchor_only`'s single-frame
case already exercised, just with a wider range), so nothing downstream needed the kind of anchor-index
offset fix the 2026-08-12 `--anchor-only` seed-prompting bug (below) needed. Verified against real
data (AIAL chain_03, anchor z=1518, `--context-frames 2`): exactly 5 frames prepared,
`{0: 1516, 1: 1517, 2: 1518, 3: 1519, 4: 1520}`, anchor at local index 2.

`docs/reference/cli.md`'s `gui.py` table also picked up `--anchor-only` and `--source`, both landed
earlier in the day but never documented there.

343 passed, 1 skipped; ruff clean; no dashes.

---

<a id="r-2026-08-12-lasso-add"></a>
## 2026-08-12, lasso-add mask tool: a freehand loop unions into the current frame's mask, undo-compatible with the paint brush

Long boundary-stretch corrections were tedious with brush-only painting: extending a mask along a
wandering membrane meant many small brush strokes instead of one gesture. Addition only, by design:
subtraction already has the eraser, and lasso plus eraser should feel like the same pairing as brush plus
eraser.

The tool is a new `self._lasso` Shapes layer, built by `_new_lasso_layer` alongside `_box` and
`_prompts` in `open_chain`. `activate_lasso_draw` arms `add_polygon_lasso`, napari's own freehand-loop
mode, so a drag needs no custom mouse handling. Two ways in: a "draw lasso (L)" dock button next to the
box button, and the `l` key. The button is the one to trust. `Shapes.class_keymap` itself holds nothing
for `l` (only `Shift`), so the shadowing does not come from a per-layer keymap at all: it comes from
napari's app-model shortcut registry (`get_settings().shortcuts.shortcuts`), which separately maps `L`
to `napari:activate_add_line_mode` on a Shapes layer and to `napari:activate_labels_picker_mode` on a
Labels layer. Plain `l` is already claimed twice over there, so the viewer-level `l` binding loses in
exactly the states you tend to be in, right after opening a chain or right after a brush correction. A
button click never enters that resolution chain.

On release, `_on_lasso_drawn` (connected to the layer's `data` event) rasterizes the finished loop with
the standalone, already-tested `_rasterize_lasso_fill` helper and writes the added pixels into the mask
through `Labels.data_setitem`, the same call the brush uses, so the stroke lands in the mask layer's own
undo history. A bounds check on the frame index sits in front of that write, since the dims slider can
run past the mask's real T extent whenever some other layer stretches it, which the lasso layer itself
briefly does while a shape sits on it.

Clearing the consumed shape has to wait a turn. A final whole-branch review caught that doing it inline
crashed every real stroke. napari's `Shapes.data` setter opens with an unconditional `_finish_drawing()`
call, so assigning `.data` from inside the `data` event that `_finish_drawing()` had itself just emitted
re-entered that same call while its shape index was already reset to `None`, raising `TypeError: list
indices must be integers or slices, not NoneType`. The mask edit had landed by then, but the drawn loop
stayed on screen and the layer was left wedged with `_is_creating` still true, which broke the following
stroke as well. `_clear_lasso` now runs on `QTimer.singleShot(0, ...)`, one event-loop turn later, after
the outer `_finish_drawing()` has returned.

`Ctrl+Z` needed a second binding to work the way this entry claims it does. `Labels.undo` is a
layer-level keybinding, so it fires only while the mask layer is active, and the lasso deliberately
leaves the Shapes layer active across consecutive strokes. A viewer-level `Control-Z` binding now calls
`self._mask.undo()` as a fallback. Layer bindings outrank viewer bindings, so with the mask layer active
napari's native undo still handles the key exactly as before, and nothing else is shadowed: neither
Shapes, Points, nor Image binds `Ctrl+Z` natively in this version.

The event filtering leans on a real napari 0.7.0 behavior: removing that last shape fires `REMOVING`
then `REMOVED`, never `ADDED`, so filtering the callback on `event.action == ActionType.ADDED` is enough
to ignore the deferred clear's own self-write, without a re-entrancy flag.

`tests/test_gui_lasso.py` now covers the event wiring, not just the pure helper. The plan had written
that wiring off as untestable without a display, which is how the crash shipped in the first place. A
`napari.components.ViewerModel` plus real Shapes and Labels layers turns out to be enough to replay the
actual draw sequence (`initiate_polygon_draw`, the drag vertices, `_finish_drawing`) with no Qt widgets,
no display, and no GPU. Five tests exercise that wiring (the crash regression itself, an out-of-range
frame, the deferred clear actually firing, and two guard-trip cases below), all failing against the
pre-fix code and passing after it. napari is not in this project's CI extras, so each of the five is
skipped there rather than failing the build; the three pure-helper tests carry no such guard and still
run everywhere.

A re-review of the fix caught two more gaps: an early-return guard (the frame bounds check, or
`self._mask`/`self.data` being `None`) returned before the deferred clear was scheduled, leaving that stroke's
loop stuck on the layer forever, which is itself self-sustaining since a stuck shape is exactly the kind
of thing that stretches the dims T-extent past the mask's own range. `_on_lasso_drawn`'s guarded body now
runs inside `try`/`finally` so the clear is always scheduled, verified with two more tests that trip each
guard and confirm the layer still ends up empty. The other gap was this doc: the how-to
(`docs/how-to/review-flagged-chains.md`) never mentioned the lasso layer, the `L` key, or the button, so
the feature existed without being discoverable; both its layer table and its keyboard-shortcut table
cover it now.

One more hardening pass: the latent-race requeue in `_clear_lasso` (a second stroke mid-flight when the
deferred clear fires) originally resubmitted itself every 0ms, which is fine for the ordinary case where
`_is_creating` clears within a turn or two, but an unrelated exception earlier in `_on_lasso_drawn` can
leave `_is_creating` stuck permanently true (napari only resets it after a successful finish), and an
unbounded 0ms resubmit against a permanently stuck flag spins the event loop forever. The requeue is now
a capped, spaced retry (50ms, 20 attempts) instead, and reads `_is_creating` through `getattr` so a future
napari rename fails toward clearing the layer rather than raising inside a timer callback.

`py -3 -m pytest -q`: 343 passed, 1 skipped. `ruff check gui.py tests/test_gui_lasso.py`: all checks
passed. The live-GUI smoke test (drag a real loop, confirm the mask extends and `Ctrl+Z` reverts it) is
still open for a human at a real display.

---

<a id="r-2026-08-12-anchor-only-seed-bug"></a>
## 2026-08-12, `--anchor-only` prompt-seeding bug: prompts/box placed off the narrowed frame, explaining a "0/56" report

Real user report opening AIAL chain_00 with `--anchor-only`: the dims slider read "0/56", every frame
past the first was blank, the seed points and box were nowhere to be seen, and a yellow skeleton dot
looked like it might mean something (it does not, `_skel` is always drawn `face_color="yellow"`, that
is just the fixed marker colour for CATMAID chain nodes, unrelated to propagation state).

All three symptoms traced to one bug in `gui.py`'s `_seed_prompts_from_state`: it placed the chain's
saved seed points and box at `self._state.anchor_frame_idx`, the RAW recorded index against the FULL
chain (56, AIAL chain_00's position in its 113-frame recording), not `self.data.anchor_idx`, the index
already correctly narrowed to 0 for `--anchor-only`'s single-frame load (the narrowing itself, added
2026-08-10, was working correctly: the on-disk `_pcrop` view really does hold exactly one JPEG).
napari's dims slider spans the largest T-extent among all loaded layers, so a Points/Shapes layer with
data sitting at t=56 stretched the slider to 56 even though the Image/Labels layers only had one real
slice, frame 0. That produced exactly what was reported: a slider reading "0/56", frames 1 through 56
blank (no image data was ever prepared for them), and the seed prompts/box invisible while parked on
frame 0 (they were real, just sitting 56 frames away). The mask shown at frame 0 was the correct
original anchor mask the whole time; only the prompt overlay and the slider range were wrong.

Fixed by reading the seed frame index from `self.data.anchor_idx` instead of the raw state value,
matching the pattern `_ensure_local_frames` already established for `anchor_idx` itself (2026-08-10).
`reset_prompts` needed no separate fix, it calls the same method. Verified: `py -3 -m pytest` still
335 passed, 1 skipped; `ruff check gui.py` clean.

A stray cache directory turned up alongside the correct one during this investigation
(`chain_views/AIAL_chain00_s8`, 113 files, legacy `_sam` naming) next to
`chain_views/AIAL_chain00_pcrop_s2` (1 file, tier-2 naming, freshly rebuilt by the anchor-only prep).
Its origin: `AIAL_chain00_s8`'s naming matches `SCALE=8`, `prepare_video_frames`'s legacy view-dir
pattern, so it dates from before `propagate_from_verified.py`/`propagate_from_corrected_seed.py`
gained their tier-2 branch (2026-08-10, same day). It is not read by any current code path
(`_ensure_local_frames` only calls the legacy branch when `crop_window` is absent, and AIAL chain_00
has one) and was not the cause of this bug. Left in place rather than deleted mid-investigation; safe
to clear manually if disk space matters.

---

<a id="r-2026-08-10-propagate-from-verified"></a>
## 2026-08-10, propagation from manually verified multi-frame seeds: mechanism verified, driver built

Student wants to test propagation quality when every conditioning mask is human-confirmed instead of
automatic, generalising the existing "manual seed-layer confirmation" roadmap lever (previously scoped
to a single anchor frame) to MULTIPLE verified frames per chain. Split into two parallel tracks: a
working copy of AIYL and AIYR's SAM3 per-slice output
(`resolution_experiments/target_perslice_only_guard_sam3_merged`) for the student to hand-correct via
`gui.py`, and the not-yet-existing pipeline support for propagating from several verified masks at once,
built in parallel.

**Working tree set up first**, so review could start immediately: copied AIYL (41 chains) and AIYR (43
chains), plus `_run_meta.json`, to `F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR`, a fresh tree,
the source per-slice output is untouched. `py -3 gui.py --output-root
"F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR" --neuron AIYL` opens it directly.

**The real mechanism checked against source before building anything.** `sam2_video_predictor.py`'s
`add_new_mask` treats any frame not yet tracked as its own conditioning frame regardless of how many
other conditioning frames already exist, and `propagate_in_video`'s default start frame is "the earliest
conditioning frame"; a frame already conditioned is returned verbatim during a sweep, never re-predicted.
Together this means a normal forward-then-reverse bidirectional sweep, rooted at whichever verified frame
is earliest, correctly threads through and preserves every OTHER verified frame it passes, and that
frame's own encoded memory feeds everything downstream of it in the same sweep direction. No new SAM2-level
mechanism was needed, only seeding more than one frame before the existing sweep, `PropagationSession`
already exposed `add_mask(frame_idx, mask)` for exactly this, just never called more than once per chain
before.

**Landed:** `pipeline.propagate.propagate_from_verified_masks(video_predictor, frames_dir,
masks_by_frame, obj_id=...)`, the multi-seed counterpart to the single-anchor `propagate()` driver.
4 real unit tests (`tests/test_propagate_from_verified_masks.py`) against a fake video predictor built
to mirror the verified real behaviour (default start frame, verbatim conditioning-frame readback),
confirming multi-frame seeding, full bidirectional coverage, and that conditioned frames read back as
the seeded mask rather than getting overwritten. Full suite: 335 passed (up from 331), 1 skipped.

**Also landed:** `experiments/propagate_from_verified.py`, a runnable driver: reads the current on-disk
masks at chosen CATMAID z's from a corrected working tree, places each into a full `_sam` canvas (via
`pipeline.chain_masks_in_sam`'s per-crop mask + offset), prepares fresh `_sam`-scale video frames
(`pipeline.crop.prepare_video_frames`), seeds all the verified frames, propagates, and writes the result
to a SEPARATE output tree (`pipeline.masks.save_masks` + a `ChainState`/`state.json`), refusing to run
if `--out` equals `--tree` so the verified-mask source can never be edited in place. Runs at legacy
full-frame `_sam` scale (`crop_anchor=False`) specifically so the verified per-slice masks and the fresh
propagation frames share one coordinate system with no remapping code needed, the same reason
`experiments/ab_seed.py` holds that scale constant when isolating a seed variable. Does not run QC
(`run_qc`)/write `qc.csv`; score the output tree afterward with `eval.merge_metric` like any other tree.

Not yet run end to end, no real corrected frames exist yet to feed it. Once the student has corrected a
few frames in `manual_verify_AIYL_AIYR`, run e.g.:

```
py -3 experiments/propagate_from_verified.py \
    --tree "F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR" \
    --neuron AIYL --chain 0 --verified-z <corrected z's> \
    --out "F:\ZhenLab\Data\output_masks\propagated_from_verified_AIYL_AIYR"
```

**Same day, real pivot after using the working tree for the first time.** Three real problems found,
each fixed before it cost more time than it already had:

**The GUI opened empty.** `sam2_utils/review_queue.py`'s `ReviewQueue` (what `gui.py` uses to build its
chain list) reads `_manifest.csv` and `_triage.csv` from the tree root; the first working-copy setup only
copied the neuron directories and `_run_meta.json`, so the queue had nothing to show, silently, not an
error. Fixed with a general, repeatable tool, `experiments/make_review_tree.py`: copies the requested
neurons' chain directories plus `_manifest.csv`/`_triage.csv` filtered to just those neurons (so the GUI's
queue is populated correctly, not referencing chains that do not exist in the copy) and `_run_meta.json`
verbatim; deliberately does NOT copy `_labels.csv`/`_review.csv` (review-state accumulators, a fresh
working copy should start a clean review session, not inherit disposition rows from whatever review
already happened on the source). Re-ran it for `manual_verify_AIYL_AIYR`, verified with
`ReviewQueue(...).refresh()` directly: 84 manifest rows (41 AIYL + 43 AIYR), matching the chain counts.

**The actual workflow needed is narrower than the multi-frame design above.** Student's real intent:
correct ONLY the anchor/seed frame per chain (not scattered frames through the chain), and the runs
happen on Narval/CCDB, not locally, propagation is too expensive to do interactively on this laptop's
GPU. This also surfaces a real technical fact worth recording, verified against the installed
`sam2_video_predictor.py` before it shaped anything: `add_new_mask` explicitly skips the memory encoder
(`run_mem_encoder=False`), and a sweep returns an already-conditioned frame's STORED output verbatim
without re-encoding it either, so a chain seeded on EVERY frame never runs the memory encoder at all,
no propagation effect happens, identical to plain per-slice. The multi-frame machinery above is not
wrong, just not what this specific workflow needs, "only fix the seed" is the single-anchor case
`pipeline.propagate.propagate()` already handled before tonight, no new pipeline code required for the
seeding mechanism itself.

**"Which chains need repropagating" also turned out to be simpler than it looked once traced
properly.** `_manifest.csv`'s status does get set to `"corrected"`, but only by `resume_propagation`
(the `G` key), which runs its OWN local video-propagation immediately, so by the time a chain shows that
status its whole tail has already changed, not just the one frame a human painted, there is no clean
record of "this single frame was hand-edited" recoverable from that signal. Since only the anchor is
ever meant to change now, the robust fix skips GUI-state entirely: `experiments/find_corrected_chains.py`
does a direct pixel comparison of each chain's anchor-frame mask, working tree vs. source tree (placed
into a full `_sam` canvas so a differing crop offset cannot produce a false negative). Real test against
`manual_verify_AIYL_AIYR`: 0 corrected chains found, correct, nothing has been touched yet.

**Landed:** `experiments/propagate_from_corrected_seed.py`, driven by that comparison, re-propagates a
chain from its corrected anchor in TWO variants for comparison (student's own idea, "for a more
comprehensive comparison"): MASK-seed (the corrected mask fed directly, `propagate(seed_mask=True,
mask_anchor=...)`) and BOX-seed (a box derived from the corrected mask via the already-existing
`box_from_mask`, fed alongside the chain's original point prompts, `propagate(seed_box=True,
seed_points=True, ...)`). Both variants reuse `propagate()` completely unchanged. Writes each variant to
its own output tree; the working tree is only ever read.

**Also landed (not yet smoke-tested on Narval):** `cluster/run_reprop_corrected_seed.sh`, a Slurm ARRAY
job following `cluster/run_exp.sh`'s exact structure (same module set, same account/gres pattern), one
task per row of `find_corrected_chains.py --out-csv`'s manifest, each task running both propagation
variants for one chain. Lint/syntax-checked locally (`bash -n`); written to match established convention
closely to minimize risk, but genuinely unverified against real Slurm/Narval from this session, needs a
one-task smoke test before trusting the full array, same discipline every other job in `cluster/` already
expects. The full round trip this is one leg of: generate per-slice on Narval (already documented, no new
script) -> pull to local (`rsync -L`) -> correct anchors in `gui.py` -> `find_corrected_chains.py` ->
push the working tree back -> this array job -> pull the two variant trees back.

**Also landed:** `experiments/report_assets.py`, headless render/metrics generation for the
"Pipeline Report" deliverable template (`neuron-gif`/`chain-gif`/`metrics` subcommands). Reuses
`sam2_utils.video_viz.to_gif`/`to_mp4`, the same functions `gui_neuron.py`'s "export overlay" button
calls, so no napari import is needed. Real end-to-end test against `manual_verify_AIYL_AIYR` chain_00
caught a real bug on the first try: rendering the whole raw `_sam` frame made a small chain's mask
imperceptible (the same "invisible mask at full-frame scale" problem this session already hit once with
a micro_sam spot-check), fixed by cropping to a fixed window covering every mask across every z in the
chain (padded), computed once so the whole render stays registered; a second pass then also raised the
default `preview_scale` from `to_gif`'s own default of 4 down to 1, since a window this small does not
need the extra downscale a full-frame render would.

**Same day, first real GUI attempt against this data surfaced two more real bugs, both fixed in
`gui.py` itself, not this session's new scripts.**

**Crash: `state.json`'s `frames_dir` is a stale Narval scratch path, and this is universal, not
specific to AIAL/AIAR.** Checked AIYL chain_00 too, same problem: `frames_dir` recorded at generation
time as `/localscratch/fsid.<jobid>.0/frames/...`, a Slurm scratch directory deleted the moment that
Narval job ended. On Windows the leading `/` also gets misread as "root of the current drive"
(`D:\localscratch\...`), so the crash is a bare `FileNotFoundError` deep inside image loading, not an
obviously-stale-path message. This also surfaced that these chains are NOT all the legacy full-frame
`_sam` case this session's new propagation scripts assumed: AIYL chain_00 itself is a tier-2 `_pcrop`
chain (`crop_scale=1`). Fixed with `gui.py`'s new `_ensure_local_frames`: checks whether the recorded
path actually exists locally, and if not, regenerates it from the source tifs via
`pipeline.prepare_chain_crop_frames` (tier-2) or `pipeline.prepare_video_frames` (legacy), whichever
`state.json`'s `crop_window` calls for. `open_chain` now loads `state.json` and resolves this BEFORE
calling `review.load_chain` (previously after), so the resolved path can be passed straight in. Real
test against the exact failing chain (AIAR chain_00, tier-2, 88 frames): confirmed missing beforehand,
regenerated in ~45s, frame 0 present afterward. The shared decode cache
(`pipeline.frames._ensure_cached_frames`) means only the first chain touching a given z-range pays this
cost again locally.

**Wrong neuron opened: `--neuron` alone (no `--chain`) was silently ignored.** `launch()`'s branch only
opened a specific chain when BOTH `--neuron` and `--chain` were given; with just `--neuron`, it fell
through to opening the first pending chain in the WHOLE tree's queue, regardless of neuron, which is
how `--neuron AIAL` opened an AIAR chain instead. Fixed: `--neuron` alone now filters the pending queue
to that neuron first, falling back to that neuron's first chain on disk if it has nothing
pending/flagged, and prints a clear message rather than silently substituting a different neuron. Real
headless test against the actual queue: unfiltered, `AIAR` chains come first (reproducing the bug
exactly); filtered to `AIAL`, correctly lands on `('AIAL', 0)`.

**That gap is now closed, same day, called out as priority #1.** `propagate_from_corrected_seed.py`
(the script actually in use for the real "only fix the seed" workflow) and `propagate_from_verified.py`
both now check `state.json`'s `crop_window` and branch to `prepare_chain_crop_frames` (tier-2) or
`prepare_video_frames` (legacy) accordingly, matching `_ensure_local_frames`'s pattern. The bigger fix
was how the corrected/verified mask itself gets read: both scripts previously went through
`chain_masks_in_sam`, which downscales a tier-2 mask to the shared `_sam` aggregation grid, fine for
cross-chain merging, wrong for feeding a mask back into a fresh propagation that now runs at the
chain's own native `_pcrop` resolution. Fixed to read the saved mask PNG directly
(`sam2_utils.qc._load_binary`), which for a tier-2 chain is already sized to the full crop window with
no offset math needed; `state.json`'s `prompts` needed no change at all, confirmed from
`pipeline/orchestrator.py`'s own comment that a tier-2 chain's seed stays in `_pcrop` space throughout
despite the "_sam" field naming. Output `state.json`s now also carry the source chain's `crop_window`
forward (previously hardcoded `None`), so a tier-2 output tree stays tier-2-aware for later reading.

Real, CPU-only verification against the exact chain that exposed this (AIYL chain_00, tier-2,
`crop_scale=1`, not run through the GPU to avoid contending with the live GUI session): the native
anchor mask reads back at shape `(1624, 1416)`, and `prepare_chain_crop_frames` independently produces
frame 0 at `(1624, 1416, 3)`, an exact match, confirming mask and frames now genuinely share one
coordinate system. `find_corrected_chains.py` needed no change, its `chain_masks_in_sam`-based
comparison is a same-space-both-sides diff either way, downscaling both trees to `_sam` equally is
fine for detecting whether something changed, just not for feeding a mask into a new propagation.

**Same day, two more real usability requests, both landed.**

**Auto-provisioning: `gui.py --source` fixes a real bug in `make_review_tree.py` along the way.**
Previously every new neuron needed a separate manual `py -3 experiments/make_review_tree.py` call
before `gui.py` could open it. `gui.py` now accepts `--source <original tree>`; combined with
`--neuron`, it auto-runs `make_review_tree` the first time that neuron is opened, and never again,
the check is a plain "does `output_root/neuron` already exist", so an already-provisioned (possibly
already corrected) neuron is never re-copied and never loses work. Building this surfaced a real bug
in `make_review_tree.py` itself: `_manifest.csv`/`_triage.csv` were being overwritten wholesale with
just the newly-requested neurons' rows, so calling it a second time for a different neuron (exactly
what repeated `gui.py --source` calls now do) silently erased every previous neuron's manifest rows.
Fixed to merge (replace only the requested neurons' rows, keep everyone else's). Real test: AIZL
(65 chains, new) added to the existing `manual_verify_AIAL_AIAR` tree (AIAL 37, AIAR 29, already
provisioned), manifest correctly reads 131 rows total, all three neurons intact, confirmed via
`ReviewQueue.refresh()` directly.

**`--anchor-only`: real 56x speedup for the "only fix the seed" workflow, and a real bug caught
building it.** The GUI's per-frame STACK loading was already fast (dask's lazy image path, confirmed
directly: ~3s to open a 40-frame stack, ~0.05s to materialize the first frame), the actual slow part
for a long chain was this same day's `_ensure_local_frames` fix decoding every OTHER frame nobody was
going to look at, especially costly for a tier-2 chain, which has no shared decode cache (every
chain's crop window is unique, unlike legacy `_sam`). `--anchor-only` narrows frame prep to just the
anchor z (`z_range` param, new on both `prepare_video_frames` and `prepare_chain_crop_frames` in
`pipeline/crop.py`, backward compatible, `None` keeps every existing caller's whole-chain behaviour
unchanged). `review.load_chain` already drops any saved mask whose z is not in the given `frame_to_z`
(sam2_utils/review.py), so the narrowing cascades into a clean single-frame session on its own, no
deeper GUI refactor needed. Real, measured: AIAR chain_00 (the same tier-2 chain that took ~45s to
fully regenerate earlier this same day) now prepares in 0.8s.

Building it caught a real bug before it shipped: the anchor's frame index has to come from the SAME
narrowed source as `frame_to_z`, not the recorded `state.json` value, an early version passed the
recorded `anchor_frame_idx` (43, AIAR chain_00's position in the full 88-frame chain) alongside the
narrowed single-frame `frame_to_z`, which would have broken every anchor-detection check downstream
against a stack that only has index 0. Fixed by returning `anchor_frame_idx` from the same prep call
that builds the narrowed `frame_to_z`, and passing both together into `review.load_chain`. Verified
directly: `anchor_idx` now reads `0`, matching the actual loaded stack. `resume_propagation` (`G`)
refuses with a clear message in `--anchor-only` mode instead of silently propagating through a
one-frame video (meaningless); the real propagation step for this workflow is the separate
`experiments/propagate_from_corrected_seed.py`/cluster job, not local resume.

---

<a id="r-2026-08-09-spill-doc-prep"></a>
## 2026-08-09, spill/overfill documentation prep: real membrane-pass cost measured, a stale CSV trap found, micro_sam and square-mask spot-checks

Prep work for the PI's spill/overfill documentation ask (roadmap item 16, §4.9): real examples of
propagation spill need both the foreign-node signal (already scored, Phase-0) and the mild-bleed
signal (the membrane pass, never run on the SAM3 propagation trees, since the 2026-08-06 full-scope
validation used `--no-membrane` for speed).

**Real per-frame membrane-pass cost measured, not assumed.** An isolated micro-benchmark of
`membrane.membrane_map` alone read ~0.14s/frame, optimistic and misleading: timing the real
`eval.merge_metric` CLI end to end (frame I/O, mask I/O, all detector passes) on 1,038 real frames
from two neurons gave 1.046s/frame, confirming the older ~0.9s/crop figure this document already
carried rather than the cheaper isolated number. At the full tree's 66,532 frames that is
~19 hours single-threaded, locally, per tree, too long for a documentation task that does not need
the whole tree.

**A stale, mismatched CSV found at `resolution_experiments/target_tier2_s1forced_neg_sam3_merged`.**
Its `_merge_metric.csv` lists 129 neurons and 66,532 frames, but only 18 neuron directories actually
exist on disk there now, 108 of the CSV's neurons are gone. The real, consistent 129-neuron,
66,532-frame full-scope propagation tree lives one level up, at
`output_masks/target_tier2_s1forced_neg_sam3_merged` (not under `resolution_experiments/`), verified
by checking the CSV's neuron set against the directory listing directly (exact match, no mismatch).
Anyone reading numbers from the `resolution_experiments/` copy of this tree should regenerate them
first; the cached file there predates whatever repurposed that directory for a smaller run and was
never cleaned up.

**`experiments/overnight_membrane_scan.py` (new): a resumable wrapper for unattended runs.**
`eval.merge_metric.score_run`'s per-chain loop has no error handling and writes its CSV once at the
end, so a single unreadable frame (F: dropped twice earlier this same evening) loses the entire run,
however many hours it had already spent. This scores one neuron at a time into its own shard CSV
(`_merge_metric.membrane_shard_<neuron>.csv`, stitchable later with `eval.concat_merge_shards
--shard-glob`), retries a failed neuron up to 3 times with a 30s backoff, and skips any neuron whose
shard already exists, so re-running the script after a crash resumes instead of redoing work.

**Scoped to the 20 highest-bleed neurons instead of the full tree**, chosen directly from the
existing Phase-0 `total_foreign` counts (`SAAVL`, `AVM`, `SIADL`, `ALA`, `AVKR`, `SIBDR`, `SMBDL`,
`RIVR`, `PVWL_or_R_3`, `SABD`, `SIADR`, `AVKL`, `SDQR`, `PVWL_or_R_1`, `AVJR`, `SIBVR`, `AVL`,
`SIAVR`, `SMDDL`, `SIAVL`), 12,067 frames, ~3.5 hours: enough real, already-known-to-bleed examples
for the documentation task, launched overnight rather than a random or full-tree scope. Smoke-tested
clean on one neuron (SDQR, 409 frames, 343s, `mild_bleed_rate=0.022`) before launching the full
batch. All 20 neurons finished clean, 0 failures, in ~63 minutes total (faster than the ~3.5h
estimate, likely because stopping the concurrent square-mask scan freed up F: I/O). Stitched with
`eval.concat_merge_shards --shard-glob "_merge_metric.membrane_shard_*.csv"` into
`_merge_metric_membrane_subset.csv`: 485 chains, 12,067 frames, `foreign_frame_rate=0.458`,
`dropout_rate=0.236`, `total_foreign=8,476`, `mild_bleed_rate=0.019`, `spanning_merge_rate=0.066`,
`boundary_on_membrane=0.927`, `underfill=0.625`. Both signals now exist together for these 20
neurons. `mild_bleed_rate` reads lower than the high `foreign_frame_rate` might suggest, expected
rather than contradictory: they measure different things, mild bleed is a boundary crossing a
membrane without reaching a foreign centreline, foreign-node containment is the severe case.

**Four real example frames rendered** (`experiments/spill_examples.py`), then substantially
revised after a real methodology problem surfaced: the first pass picked the single worst FRAME
by foreign-node count per example and paired it with "the next chronological frame" for a fourth
panel. Direct feedback that two of those pairs "were already problematic" led to actually checking
each example chain's full timeline from its own anchor (`anchor_catmaid_z`, read straight from
`state.json`) outward, which found the real reason: all three original picks
(`SAAVL chain_17`, `SMDDL chain_03`, `AVJR chain_03`) already had a foreign node AT THE ANCHOR
FRAME ITSELF. There was no clean-to-bled transition to show, because none of them ever started
clean, "the next frame" was just more of the same by construction.

That distinction became the real finding. The examples now split into two real kinds, found by
checking data, not by assumption:

- **`bad_seed`** (the anchor itself is already contaminated): `SAAVL chain_17`, anchor z=1460,
  already touches 1 foreign node there; 108 frames out at z=1568 the mask has grown into a long
  band along what reads as body wall muscle, engulfing 9 foreign nodes and losing its own node
  entirely (`own_contained=False`, confirmed against the CSV). A small seed-level contamination
  compounding into a severe, own-target-losing chronic bleed. `AVJR chain_03`, anchor z=1610, is
  worse at the seed itself (6 foreign nodes already) and stays in the same range (8 at z=1607), a
  mask that was never on-target rather than one that drifted there.
- **`drift`** (the anchor is genuinely clean): real search across the 20-neuron subset's full
  per-chain timelines found 137 chains with a clean anchor (own node contained, zero foreign
  nodes) that bleed later. `SABD chain_25` (anchor z=1407) stays completely clean for 77
  propagated frames before its first foreign-node hit at z=1330; `SIADR chain_06` (anchor z=1404)
  stays clean for 23 frames before onset at z=1427. Both render as a genuine transition pair, last
  clean frame vs. first bled frame: a tight mask inside a real, clearly bounded compartment, then
  a small bulge crossing a real (often locally weaker) membrane boundary into the neighbour. This
  is the clean "propagation caused this" story the `bad_seed` chains cannot tell.

**Two rendering bugs found and fixed while building these, both real, not cosmetic.** First, node
markers only drew a green star for an own node found INSIDE the mask, but `score_chain`'s actual
own_contained logic checks ANY of a neuron's own nodes at that z (a neuron can have more than one
branch point sharing a z, e.g. AVJR had three distinct nodes at z=1607); the fix draws every own
node, hollow when the mask misses it, filled when contained, which is what surfaced the SAAVL
own-target-loss finding above, an earlier version would have shown nothing there at all. Second,
letting an own node's position expand the crop window blew the window out to include an unrelated
distant branch, shrinking the actual mask to an unreadable speck; fixed to size the window from
the mask and foreign nodes only, with the axes limits explicitly locked before any off-window node
is scattered (an unlocked matplotlib scatter call was silently auto-expanding the whole view to
fit the far-away point, the same shrink-to-speck symptom from a second, unrelated cause).

Still open: a systematic survey beyond these four (137 real `drift` candidates exist to pick from
if more are wanted), and how common `bad_seed` is relative to `drift` across the tree as a whole,
not established by four hand-picked chains.

Stitching hit a real bug in `eval.concat_merge_shards`: its shard-sort key assumed the cluster's
numeric shard index (`shard_<i>.csv`) and crashed with a `ValueError` on this driver's
neuron-named shards (`membrane_shard_SDQR.csv`). Fixed with a sort key that falls back to
alphabetical ordering for a non-numeric suffix (`_shard_sort_key`, `eval/concat_merge_shards.py`);
ordering is cosmetic either way, concatenation is a plain row union. Full test suite still passes
(331 passed, 1 skipped), and `tests/test_concat_merge_shards.py` specifically still passes (7/7).

**Narval/CCDB considered and set aside for this round.** No cluster script runs `eval.merge_metric`
today, only mask generation (`cluster/run_array.sh`); building one, uploading these mask trees to
project storage, and working around the precedent of the `retro_sam3` cluster scoring job's earlier
12-hour timeout would cost more setup time than the ~19 hours it would save on a one-off, and the
targeted local subset above sidesteps needing the full tree at all. Worth revisiting if this becomes
a repeated need.

**`experiments/microsam_spotcheck.py` (new): micro_sam as a zero-shot comparison baseline.** Same
single node-point prompt fed to both our SAM2 pipeline and micro_sam's `vit_l_em_organelles`
checkpoint (the only EM-domain micro_sam checkpoint that actually exists in the installed registry;
an earlier assumption of an `em_boundaries` checkpoint, based on micro_sam's own documentation, does
not match `micro_sam.util.models()`'s real contents, checked directly). One frame (AIYL, z=1456):
our SAM2 mask landed on the prompt point as expected; micro_sam's mask did not, it captured an
unrelated organelle-like blob well away from the point. Not a verdict from n=1, but consistent with
this document's existing prediction that an organelle-trained checkpoint would struggle on a neurite
prompt, and a sharper failure than expected, the prompt barely seemed to matter. Getting there also
surfaced a real, unrelated bug: the installed `segment_anything` predictor's `.cpu().numpy()` call
has no bfloat16 support, and this GPU runs the checkpoint in bfloat16; forcing the micro_sam
predictor onto CPU sidesteps it for a one-frame spot-check.

**Square-mask ("actual squares filling their crop") check re-run with the corrected metric.**
`check_square_masks.py` (session scratchpad) measures `mask.sum() / mask.size`, the mask's own
assigned crop window, not its tight bounding box (an earlier version of this check measured the
wrong ratio against the mask's own tight bounding box instead, which reads high for any compact
blob and does not distinguish a real square-crop-fill from an ordinary round mask). Real interim
result: 0 of 25,735 masks in the propagation tree hit `crop_fill >= 0.9`, no
evidence of literal square-filled-crop masks there under this metric. The secondpass tree's scan is
still incomplete, interrupted twice by F: dropping mid-scan.

---

<a id="r-2026-08-06-second-pass-fullscope"></a>
## 2026-08-06, propagation second pass: full-scope validation, a real verdict

Retro-scored the plain-propagation and propagation+second-pass trees fresh (Phase-0,
`eval.merge_metric --no-membrane`), full 133-neuron target scope on both, identical 4,041 chains
and 66,532 frames, a clean apples-to-apples comparison rather than the one curated chain the
2026-07-30 landing verified. Bleed rate 27.2% to 17.4% (-36% relative), dropout rate 14.0% to 2.7%
(-81% relative), total foreign nodes 23,993 to 19,537 (-18.6%), mean z-to-z IoU 0.681 to 0.668
(flat, the temporal-consistency advantage is not traded away for this). Substantially stronger than
the one-chain spot check suggested, and it holds despite the still-open accept-gate gap from that
landing (the gate only checks the re-predicted mask contains its own node, not that the specific
foreign node which triggered the flag is gone), so the true improvement is likely larger than these
numbers, not smaller. The propagation second-pass feature now reads as a validated win at full
scope, not an uncalibrated first signal. Also surfaced along the way: the full-scope
plain-propagation bleed rate (27.2%) is meaningfully higher than the 21.7% quoted elsewhere in this
document and the presentation deck, which came from a curated 15-neuron matched-comparison subset;
not a regression, just a different, harder, more representative population, the two numbers measure
different things and should not be compared as before/after.

Also hit two real problems getting to this data. `cluster/stage_download.sh`'s idempotent skip
check was presence-only (`-f`, not integrity-checked), so a tarball truncated by an earlier timeout
would be silently trusted on the next run; fixed to verify with `tar -tzf` and retar on failure
(caught two previously-unknown corrupt archives on its first real run under the fix, not just the
one known-truncated tree, see the 2026-08-04 entry below). Separately, local extraction of the two
big target trees hit F:'s known physical flakiness mid-write ("No such device"), unrelated to the
tar fix; resolved by reconnecting the drive and retrying cleanly, with the partial extraction
removed first so the retry was not overlaying stale files.

---

<a id="r-2026-08-04-mitonet"></a>
## 2026-08-04, NucleoNet recall gap found, MitoNet mitochondria detector spot-checked

A human visual re-check of the NucleoNet renders from the 2026-07-29 gate run (z=1456, z=1472)
confirmed the "no false positives" verdict but surfaced a gap the original writeup did not carry:
z=1456 has at least one visible nucleus NucleoNet did not detect. The GATE VERDICT stands (NucleoNet
generalizes, no false positives on organelles), but the roadmap's NucleoNet section
(`docs/explanation/roadmap.md` §4.5) now carries an explicit recall caveat: this detector can miss
real nuclei, not just avoid flagging non-nuclei, worth remembering before treating its output as a
complete nucleus census for anything downstream.

Separately, a naive round-plus-dark-mask heuristic proposed as a possible detector for organelle
capture turned out, on clarification, to target mitochondria rather than nuclei: small, dark,
round-to-elongated blobs, the same class of structure the membrane ridge map already misfires on
(`sam2_utils/membrane.py`, see the 2026-07-29 organelle blob suppression entry below). Rather than
build a circularity/intensity heuristic from scratch, this session checked whether a pretrained
mitochondria detector already exists in the same registry NucleoNet came from. It does:
`empanada-napari` ships `MitoNet_v1`, trained on the CEM-MitoLab dataset, config confirmed against
`empanada_napari/configs/MitoNet_v1.yaml` (fetched 2026-08-04), which also closes part of the
NucleoNet provenance gap by confirming `nms_threshold`/`nms_kernel`/`confidence_thr` are real
training-time values, not inference-engine defaults carried over unmodified.
`experiments/mitonet_spotcheck.py` (new, mirrors `nucleonet_spotcheck.py`'s vendoring approach) ran
on the same two target-worm frames used for NucleoNet: 74 instances detected at z=1456, 71 at
z=1472, both plausible mitochondria counts for a worm cross-section. A human visual check of both
renders found the same pattern as NucleoNet: accurate on what it catches, incomplete on what it
misses. GATE VERDICT: MitoNet generalizes for detection. Of the candidate uses raised (ridge-map
suppression, SAM-mask mitochondrion-capture detection, a general organelle classifier), ridge-map
suppression was picked first since it is the one with a documented open failure: the 2026-07-29
organelle blob suppression entry below found the classical Otsu/shape detector calibrated to ~96%
single-pixel noise and moved the real bleed floor by exactly zero. See roadmap item 2f (§4.5).

`experiments/organelle_pretrained.py` (new) unions MitoNet's and NucleoNet's full-frame instance
masks into one boolean mask (both organelle types by default, one forward pass each per frame, not
per neuron crop) and `experiments/dense_membrane_fill.py` gained `--organelle-source
{classical,pretrained}` so `grow_all` can slice a neuron's region out of that precomputed mask
instead of calling `detect_organelle_blobs` on its local crop. Same real gate as the classical run
(z=1456, 117 neurons, `new_bleed` = cells that only start engulfing a foreign node after being
grown): baseline (no suppression) and the classical detector both land on `new_bleed=7` at the
canonical `uf_min=0.6, cap=5x` gate; the pretrained mask (3.2% of the frame flagged, vs. the
classical detector's much noisier ~15% per-crop footprint) lands on `new_bleed=6`. The full
`uf_min` sweep (`0.00`/`0.40`/`0.50`/`0.60`/`0.70`) shows the same story: baseline 34/14/11/7/3
vs. pretrained 33/14/11/6/3, a one-cell dent at best, zero at most gates. This is a materially
different result from the classical detector's flat zero, since it is a semantically real
organelle mask rather than 96% Otsu noise, and it still barely moves the floor. That points away
from organelle contamination of the ridge map as the dominant driver of `new_bleed` and back toward
the scale-8 leaky-wall problem already flagged by the 2026-07-27 dense-frame membrane-fill sweep
(the flat ~40% bleed-per-fill rate, attributed there to the membrane map's resolution, not
organelle noise).

---

<a id="r-2026-07-30-second-pass"></a>
## 2026-07-30, propagation second pass: neighbour-seeded re-segmentation lands, SAM3 crash found and fixed, accept-gate gap flagged

Phase 1.5's re-anchoring lever. A finished propagation chain's flagged frames (dropout or foreign-node
bleed, from `eval.merge_metric.score_chain`) now get a targeted re-segmentation pass instead of being
discarded or forcing a full chain re-run: `select_second_pass_frames` and `find_second_pass_neighbour`
(`pipeline/propagate.py`) find the flagged z's and their nearest unflagged neighbour in the same
chain, and `apply_second_pass` re-predicts each flagged frame from that neighbour's mask plus the
frame's own skeleton-node point, falling back to `apply_blowup_guard`'s existing neighbour-copy when
the result fails a sanity check or no unflagged neighbour exists. This needed a new `mask_input`
parameter on `image_predict` (`pipeline/predict.py`) and a `mask_to_low_res_logits` helper to convert
a saved boolean mask into SAM2's expected low-res logits. Both gated off by default:
`PipelineConfig.second_pass: bool = False` and `second_pass_min_neighbour_area_ratio: float = 0.5`,
driven by new `batch.py --second-pass`/`--second-pass-min-neighbour-area-ratio` flags, folded into
each chain's `qc.csv` as a `second_pass` column recording which tag applied, `"corrected"` or
`"guard_fallback"`. Because `pipeline/propagate.py` is inside the
enforced library boundary and cannot import `eval`, the trigger's `score_chain` call lives in
`batch.py` instead, with the resulting records passed into `apply_second_pass` as plain dicts. Full
design, including the diagnostic that motivated it (propagation's `foreign_frame_rate=0.252` against
SAM3 per-slice's `0.087`, same backend, per ADR 0017) and the nucleus-capture neighbour guard's
stated limits (it only helps when nucleus-capture affects a minority of a chain's frames, and cannot
catch a structurally trapped point prompt), see
`docs/superpowers/specs/2026-07-30-propagation-second-pass-design.md`.

Real verification against one chain, `AIZL/chain_51` (`target_tier2_s1forced_neg_sam3_merged`, 11
frames, 5 flagged), turned up two findings. First, a real crash: `apply_second_pass` raised
`TypeError` against the real `Sam3ImagePredictor`, whose `predict()` has no `mask_input` parameter at
all, which on a real `--second-pass --backend sam3` run would mark every touched chain `FAILED` after
a successful propagation and force a full re-propagation on resume. Fixed (commit `802669e`):
`image_predict` now omits `mask_input` unless it is not `None`, and `apply_second_pass` checks the
predictor's signature once and falls back to a point-only re-predict on any backend without the
parameter. Verified against the real predictor on local GPU; SAM2's path is unchanged. Second, an
unfixed gap: the accept/reject gate only checks that the re-predicted mask contains the frame's own
node, never whether the specific foreign node that triggered a bleed flag is gone. On the verification
chain's 5 touched frames, only 2 genuinely resolved their trigger; the other 3 were tagged
`"corrected"` while `n_foreign` stayed unchanged at the same foreign node id.
A `"corrected"` count should not be read as a "fixed" count without this caveat, and it remains open
for a future iteration. One small chain: a first signal on both fronts, not a verdict, though the
crash itself is not a one-chain fluke, it follows directly from the SAM3 adapter's signature. See
`.git/sdd/task-5-report.md` for the full before/after tables and the roadmap's item 13 (§5b).

The final whole-plan review (commit `306da30`) caught a third, more basic problem the real-chain
verification above could not see, since it always ran against a chain whose `state.json` was already
correct on disk: `_apply_second_pass_and_update_qc` was called before `save_state`, so on a chain's
first real run its internal `score_chain` call read a missing or stale `state.json`, mis-learned the
chain's crop space, and flagged every frame with dozens of phantom foreign nodes, which meant no
frame ever had an eligible neighbour and the whole pass silently did nothing. Fixed by moving the
call to after `save_state`. The same review also reversed the `"corrected"` clear-flags behavior
described above: given the accept-gate gap just above, clearing a frame's triage flags on an
unverified `"corrected"` tag was actively hiding still-broken frames from review, so both
`"corrected"` and `"guard_fallback"` now force `flag`/`intervene`/`queue` back to `True`, the
`second_pass` column still records which tag applied. A failure inside the pass is now also isolated
in a `try/except`, so it cannot fail an already-successful chain.

---

<a id="r-2026-07-29-organelle-blob"></a>
## 2026-07-29, organelle blob suppression: the intensity/texture filter lands, calibrated against real EM, the floor still does not move

Phase 2b.5's second classical lever, tried after temporal projection landed as a negative result.
`sam2_utils/membrane.py` gained `detect_organelle_blobs` and `suppress_organelles` (Otsu threshold,
connected-component labeling, then `regionprops` area/eccentricity filtering, with a dilation margin
before inpainting), wired into `experiments/dense_membrane_fill.py`'s `grow_all` via
`--suppress-organelles`/`--blob-max-area`/`--blob-max-eccentricity`/`--blob-dilate-px`. An earlier
`blob_dog` (Gaussian-scale blob detector) design was rejected before this plan existed, verified
directly to not tell blob shape from ridge shape (it flagged a synthetic ridge about as strongly as a
synthetic blob), which is why the shipped mechanism filters on `regionprops` shape statistics instead.

Calibration against a real target-worm EM crop (z=1456, 200x200px, Otsu threshold 122.1, 43.1% of the
crop dark) found the two shape knobs cannot do what real data needs from them. Of 1017 dark connected
components, 710 pass the plan's default filter (`max_area=150.0`, `max_eccentricity=0.85`), and 618 of
those (87%) sit at area <= 3px, near-certain single-pixel Otsu noise rather than organelles; only 28
(4%) fall in a plausible organelle size range (area 11-144px). Tightening `max_eccentricity` to 0.6
does not fix this (562 of 585 kept components, still 96%, stay <= 3px) while discarding 90% of the
plausible-organelle-sized candidates, a net loss, not an improvement. Tightening `max_area` to 60-80
barely changes the kept count (706-708 vs 710), so area is not the binding constraint either; both
shape knobs stayed at the plan's default for these reasons. `blob_dilate_px` is what actually
controls collateral damage, since dilation lands on hundreds of noise specks, not the ~28 real ones:
the flagged-pixel fraction of the crop runs 5.3% at `dilate_px=0`, 14.6% at `dilate_px=1`, 26.1% at
`dilate_px=2`. `blob_dilate_px` was lowered from the plan's default of 2 to 1, halving the
over-suppression footprint to 14.6% of the crop, a mitigation for the noise-speck problem rather than
a fix for it (the current interface has no minimum-area floor to remove single-pixel specks
directly).

The calibrated gate (`py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6
--suppress-organelles --blob-max-area 150.0 --blob-max-eccentricity 0.85 --blob-dilate-px 1`) reported
filled=18, foreign=40, bleed_cells=28/117, new_bleed=7, mean_uf=0.381, area +12%, contested=3274,
against a freshly reproduced same-day baseline at the same z and `uf_min` (no suppression) of
filled=17, foreign=39, bleed_cells=28/117, new_bleed=7, mean_uf=0.403, area +12%, contested=2749. That
reproduction mattered: the 0.38 bleed-per-fill figure already on record from the temporal-projection
entry below turned out, on closer reading, to belong to a different setting (the `uf_min=0` run), not
this `uf_min=0.6` run, so it was not the correct baseline to compare against. Against the correct,
apples-to-apples baseline, the rate moves from 7/17=0.41 to 7/18=0.39; against the older 0.38 figure it
reads flat at 7/18=0.39. Either way, `new_bleed` itself is identically 7 in both conditions; the only
thing that changed is the denominator, one additional cell crossed the `uf_min>=0.6` grow-eligibility
gate under the suppressed membrane map, which is not the same as any grow leaking less. Contested
pixels (mask overlap between neurons) rose 2749 -> 3274, +19%, a cost the bleed-per-fill metric does
not capture.

**A whole-plan review pointed out that this single `uf_min=0.6` row (n=17/18 filled) is the
narrowest, most favorable slice available, and that `--sweep` already prints a much larger,
population-matched comparison across every `uf_min` gate from the same grow pass, for free.**
Reproduced directly (`py -3 experiments/dense_membrane_fill.py --z 1456 --sweep`, baseline vs
`--sweep --suppress-organelles --blob-max-area 150.0 --blob-max-eccentricity 0.85
--blob-dilate-px 1`):

| uf_min | baseline new_bleed/filled | suppressed new_bleed/filled |
|---|---|---|
| 0.00 | 34/90 = 0.378 | 35/90 = 0.389 |
| 0.40 | 14/36 = 0.389 | 15/34 = 0.441 |
| 0.50 | 11/26 = 0.423 | 11/24 = 0.458 |
| 0.60 | 7/17 = 0.412 | 7/18 = 0.389 |
| 0.70 | 3/11 = 0.273 | 4/13 = 0.308 |

At `uf_min=0` the grown population is perfectly matched (filled=90 on both sides, 5x the sample of
the `uf_min=0.6` row). Most tracked stats are worse under suppression at that row: foreign 139 ->
142, bleed_cells 55/117 -> 56/117, contested 25741 -> 28107 (+9%), area +86% -> +89%. `mean_uf` is
the one stat that improves (0.313 -> 0.293), the ordinary underfill/bleed tradeoff: suppression lets
a few more cells clear the fill gate, which lowers mean underfill while raising bleed and contested
overlap. Suppression is flat-to-slightly-worse at 4 of the 5 `new_bleed/filled` gates; `uf_min=0.6`
is the sole exception, and it is also the smallest sample in the table, so it reads as the
coincidence of one small row rather than the representative case.

**Result: the floor does not move, confirmed across a 5x larger, population-matched sample, not just
the narrow n=18 row first reported.** This is the plan's second documented negative outcome (temporal
projection was the first), and the population-matched check strengthens rather than overturns it.
This closes out the specific `detect_organelle_blobs`/`suppress_organelles` design landed in Task 1,
not organelle suppression as a general idea: the current detector has no minimum-area floor, so it
structurally cannot separate single-pixel Otsu noise from genuine tiny organelle fragments, and a
differently-designed detector (an explicit minimum-area floor, or local rather than global Otsu
thresholding) remains a distinct, unexplored option. With both classical levers now tried and
measured negative, items 2c (grow-to-membrane) and 2d (non-overlap arbitration) stay gated, and per
the roadmap's own decision point, the next step is a learned membrane map (Phase 3), judged on
boundary sharpness rather than zoomed-out neatness. See the roadmap's item 2b.5 (section 5), item 10,
and the decision point in section 6.

---

<a id="r-2026-07-29-z2z"></a>
## 2026-07-29, z-to-z consistency metric: a temporal ruler for the merge metric's per-frame blind spot

The target-worm merge metric (`eval/merge_metric.py`) scores each frame independently, so it has no
way to see whether a run's masks hold their shape from slice to slice, and it structurally favours
per-slice methods over video propagation, the method that buys that consistency. Phase 0.a closes
that gap.

`z_transitions` computes IoU and centroid drift between every pair of a chain's adjacent-z masks, in a
shared coordinate frame built from each mask's own `_sam`-grid offset (a tier-2 crop window can move
or resize between frames, so the two local arrays cannot be compared directly). A transition touching
an empty mask gets `iou=None`, kept separate from the low-consistency signal so a chain that scores
well only because it dropped frames does not look falsely consistent. `summarize_z_consistency`
aggregates these into `mean_z2z_iou`, `mean_centroid_drift_px`, `frac_gap1_transitions`
(consecutive-slice transitions versus ones spanning a missed frame), `frac_low_iou` (the fraction of
gap-1, scored (non-dropout) transitions scoring below a new `--low-iou-threshold`, default 0.5), and
`n_dropout_transitions`. Both functions are wired into `score_run` and `format_summary`'s summary
line, and `score_run` now writes a `_z_consistency.csv` per run tree (one row per transition)
alongside the existing `_merge_metric.csv`.

A real smoke test against `original_perslice_only_guard_merged` (629 chains, 8052 frames) returned
`mean_z2z_iou=0.592 mean_centroid_drift_px=5.45 frac_low_iou=0.285 frac_gap1=1.000`, the first number
for something the project previously only had a visual read on (the presentation deck's
`perslice-jitter.png` showed the jaggedness informally, without a number attached).
Retro-scoring specific trees for an actual per-slice-vs-propagation comparison is a follow-on use of
this tool, not part of this landing; see the roadmap's item 12 (§5, Phase 0.a).

---

<a id="r-2026-07-29-nucleus"></a>
## 2026-07-29, nucleus-capture detector: GUI label added, NucleoNet spot-checked and verdicted

Phase 2e, the nested-membrane ceiling's nucleus-detection lever, got a design, a spot-check, and a
docs update in one round. The classical fallback was skipped on purpose, per the design's gate below,
not left undone.

**Task 1: the `"nucleus"` GUI label.** `sam2_utils/labels.py`'s `ERROR_TYPES` gained `"nucleus"`
(`("wrong_object", "under", "over", "bleed", "fragmented", "missing", "nucleus", "other")`), so the
napari review GUI's existing error-type dropdown (`gui.py:1127`, which reads the tuple directly, no
new UI code needed) can now tag nucleus-capture cases during normal review. That is how the labeled
ground truth that formal scoring will eventually need starts to accumulate, as a byproduct of
normal review rather than a dedicated labeling pass.

**Task 2: NucleoNet spot-check, GATE VERDICT "generalizes."** NucleoNet turned out to be real, not the
unconfirmed name it started as: a BSD-3-Clause-licensed, pretrained EM nucleus instance-segmentation
model (Panoptic DeepLab architecture, bioRxiv preprint April 2026, `volume-em/empanada`/
`empanada-napari`). Its headless inference API took real discovery work: the PyPI `empanada-dl`
package ships the model architectures and the inference engine but no model zoo. NucleoNet's
checkpoint URL and config (`NucleoNet_base_v2.yaml`) live only in the `empanada-napari` GitHub repo,
so two small napari-free helper functions were vendored from `empanada_napari/utils.py` into
`experiments/nucleonet_spotcheck.py` (a BSD-3-Clause attribution comment travels with them), sparing a
full GUI/Qt install just to reuse two functions.

Run on two target-worm EM frames: z=1456 (2 instances detected) and z=1472 (5 instances detected).
Both the implementer and the controller read the rendered overlays independently: detections landed
on round, membrane-bound structures with a mottled interior texture clearly distinct from the
surrounding organelle-packed cytoplasm, with tight mask boundaries, and at z=1472 several of the
detections were already anticipatable by eye from the raw EM panel before the overlay was applied. No
false positives on mitochondria or other organelles in either frame. Per the design's gate, this
verdict skips the classical dark-blob/thick-loop fallback entirely; `sam2_utils/nucleus.py` and
`tests/test_nucleus.py` were not built.

The rendered evidence lives at `docs/figures/presentation/nucleonet-spotcheck/`; regenerate the two
full-frame overlays with `py -3 experiments/nucleonet_spotcheck.py --z 1456` and `--z 1472`. That
folder also has a `zoom_check.png` close-up crop used for the visual read above; it was made ad hoc
and its generating code was never committed, so only the two full-frame overlays are reproducible.

One follow-up surfaced during task review and fixed in a separate commit: the inference engine's
`nms_threshold`/`nms_kernel`/`confidence_thr`/`coarse_boundaries` kwargs have unconfirmed provenance,
unlike the `NUCLEONET_*` constants, which each carry a source URL and fetch date. Now documented as a
comment at `experiments/nucleonet_spotcheck.py:124-129` for whoever revisits this detector.

**Task 3, skipped.** The plan explicitly skips the classical fallback when the NucleoNet gate says
"generalizes," so no `sam2_utils/nucleus.py` detector was built this round.

This is a qualitative, visual-only read; no quantitative precision/recall exists yet, and formal
scoring is blocked on the labeled set Task 1's GUI change starts collecting. Wiring NucleoNet into
`multimask_generous` or any other live pipeline lever stays a separate, future spec, matching the
design's stated scope. See the roadmap's item 2e and
`docs/superpowers/specs/2026-07-29-nucleus-detector-design.md`.

---

<a id="r-2026-07-29-temporal"></a>
## 2026-07-29, temporal membrane projection: register/project crops land, a gate confound found and corrected

`sam2_utils/membrane.py` gained `register_crops` and `project_crops` (Task 1), and
`experiments/dense_membrane_fill.py` gained `--mm-window`/`--mm-combine` to route a small window of
adjacent z-slices through them before the ridge filter (Task 2), plus a `--sweep-temporal` flag that
grids `window in {0, 1, 2}` x `combine in {median, mean, max}` and prints the same bleed/underfill
table as `--sweep`, now with a `filled` population column (Task 3, extended during a final whole-plan
review). `--mm-window 0`, the default, is verified byte-identical to the pre-existing single-slice
path. The review also caught a real bug in `register_crops` (it picked `crops[len(crops) // 2]` as the
alignment reference even when the caller's true center crop was elsewhere, e.g. after a frame load
failed near the stack edge) and two missing files (`experiments/dense_overlay.py` and
`experiments/membrane_autofill_demo.py`, both untracked dependencies of `dense_membrane_fill.py`); both
are fixed and committed separately.

The first gate run (`--z 1456 --uf-min 0.6 --sweep-temporal`, 117 neurons) answered item 2b.5's open
question, does averaging organelles out across z drop the ~40% bleed-per-fill floor found in the
2026-07-27 dense-frame sweep, and reported every non-baseline setting worse than window=0 (foreign 39,
bleed_cells 28/117, mean_uf 0.403, area +12%) on every axis at once: window=1 roughly doubling
foreign-node bleed (76-82, bleed_cells 39-41/117), window=2 worse again (foreign 110-124, bleed_cells
42-54/117). The write-up called this a straight regression.

The whole-plan review flagged that conclusion as under-supported: the 0.6 underfill threshold is
compared against EACH row's own membrane map, so a blurrier temporal map reads as more underfilled even
where a mask did not really change, and the fixed threshold quietly pulled more cells into the grow
step as the window widened (`filled`, now printed in the table, rises from 17/117 at window=0 to
40-49/117 at window=2). That alone could explain most of the apparent regression, independent of
whether any individual grow got worse.

A second gate run (`--z 1456 --uf-min 0 --sweep-temporal`) grows every underfill-eligible cell
regardless of window, holding the grown population much closer to constant. `filled` still moves, from
90/117 at window=0 down to 61-77/117 as the window widens, but in the opposite direction from the
first run's confound: at uf_min 0 the only thing that can drop a cell out of `filled` is the 5x
runaway-area cap, so this is the cap reverting more grows as the window widens (27/117 at window=0
rising to up to 56/117 at window=2), itself evidence the membrane wall is weakening under temporal
projection, not just a caveat about population control.

Absolute foreign-node counts under this fairer test look mixed: window=1 reads close to flat (foreign
136-148 vs baseline's 139, bleed_cells 54-55/117 vs 55/117) despite growing fewer cells, window=2 is
moderately worse (foreign 161-189, bleed_cells up to 61/117). But the clean readout is the bleed-per-fill
rate `filled` was added to compute, `new_bleed / filled`: baseline 34/90 = 0.38, window=1
33-34/63-77 = 0.43-0.52, window=2 31-41/61-73 = 0.46-0.67. Every non-baseline setting is worse per fill
than baseline with no exceptions, unlike the absolute counts. Window=2's regression cannot be explained
by gate membership (it shows up with fewer cells grown than baseline) and is a real per-fill effect;
window=1's near-flat absolute count still costs a worse per-fill rate, so the first write-up's "roughly
doubles" claim does not survive population control, but neither does "no worse."

A shift-clamp diagnostic was also added to `grow_all` (prints once per `grow_all` call when
window > 0): `register_crops` clamps every estimated shift to +/- 5px before applying it, and the
diagnostic logs the raw, pre-clamp shift magnitude for every crop pair. At window=1, 32 of 234 crop
pairs (14%) had a raw shift larger than the clamp, with raw magnitudes up to 173px; at window=2, 139 of
468 (30%) were clamped, up to 269px. Those magnitudes are far beyond any plausible real slice-to-slice
jitter at scale 8, so they most likely mean phase correlation locked onto a spurious peak on some crops
rather than found real drift, and the clamp then truncates and applies that bogus shift rather than
skipping the crop, actively misaligning it before the combine step. Misregistration via the clamp is
therefore a demonstrated contributor, not the "plausible but unverified" blur-only guess from the first
write-up. Blur from combining otherwise-aligned crops may still also play a part, the evidence here does
not separate the two mechanisms cleanly, but the clamped fraction roughly doubling from window=1 to
window=2 lines up with where the regression is worst. The frame-loading loops now also print which z's
actually loaded vs were skipped, so a degraded window shows up in the log instead of silently changing
what the sweep measured.

Item 2b.5 stays open, and 2c/2d/2e stay gated on it, now pointed at the deferred intensity/texture blob
filter (2e's nucleus-detection idea, out of scope for this plan) instead of the temporal lever, since no
setting improves the bleed-per-fill rate even where an absolute count happens to land at or slightly
below baseline by growing fewer cells, under the fairer,
population-controlled test. One untested, cheap follow-up worth a look before writing off the lever
entirely: reject spurious large-shift crops instead of clamping and applying them. See
[[membrane-temporal-projection-idea]] and the roadmap's item 2b.5 and queue item 10 for the recorded
outcome.

---

<a id="r-2026-07-29"></a>
## 2026-07-29, SAM3-vs-SAM2 scorecard consolidated, default backend decided

The Globus download of the Narval trees (both production SAM3 configs, the four-way neg x gen A/B,
and the matching SAM2 baselines) landed on F:. Each tree already carried a stitched
`_merge_metric.csv` from the sharded eval, so the comparison table did not need the timed-out
`retro_sam3` cluster job at all: `eval.merge_metric.summarize()` run directly on the cached CSVs
produced the same numbers in seconds, entirely locally, no mask I/O.

SAM3 `perslice_only_guard` wins the trusted severe-bleed ruler over SAM2 `perslice_only_guard` on
both frequency (foreign_frame_rate 0.087 vs 0.109) and severity (a bleeding frame touches 1.17
foreign skeleton nodes on average for SAM3 vs 5.45 for SAM2). The 2x2 neg x gen A/B refutes the
roadmap's pre-registered guess that negatives would hurt SAM3: negatives still help on every axis,
generous still hurts, so the existing `original_perslice_only_guard` preset is already the best SAM3
cell and needs no retune.

Decision, recorded in [ADR 0017](adr/0017-sam3-scorecard-and-default-backend.md): `--backend sam3`
stays opt-in rather than becoming every preset's silent default, because SAM3's underfill is still
notably higher (1.09 vs 0.62, the known tight-mask cost Phase 2c is meant to fix) and the ~2.2-2.5x
compute cost is a call worth leaving explicit per run rather than folding into the default. Phase 1.a
is closed. `sam3_fullres` (partial out-of-memory, unmerged) stays open only if full-resolution SAM3
is wanted later.

---

<a id="r-2026-07-23"></a>
## 2026-07-23, sharded parallel scoring, the SAM3 config A/B presets, and a documentation sweep

Follow-on plumbing for the SAM3 whole-set round, plus a documentation reconciliation.

**Sharded, parallel scoring.** `eval.merge_metric` used to score a whole tree on one core. It now
takes `--neurons` (a comma- or space-separated subset) and `--out-csv` (a per-shard destination), so
`cluster/run_eval_array.sh` splits one merged tree across CPU array tasks, each scoring its own
neuron(s) into `<tree>/_merge_metric.shard_<i>.csv`. `eval.concat_merge_shards` then stitches those
shards into the canonical `<tree>/_merge_metric.csv`, identical to what a single-core run writes, and
prints the whole-tree summary. `cluster/run_py.sh` runs any CPU step (merge, concat, retro-eval) as a
dependency job so nothing heavy lands on the login node, and `cluster/submit_sam3_round.sh` queues the
whole SAM3 test round (segmentation, merge, score, concat, retro-eval) in one shot with `afterok`
dependencies.

**SAM3 config A/B presets.** Two `k_max_neg = 0` presets, `original_perslice_only_guard_kneg0` and
`original_perslice_guard_kneg0`, complete a 2x2 sweep of `k_max_neg` in `{0, 3}` by
`multimask_generous` in `{off, on}` run with `--backend sam3`, to test whether a more conservative
model does better with negatives switched off. The two `k_max_neg = 3` corners are the existing
per-slice guard presets. `k_max_neg = 0` removes negatives entirely (build_prompts caps them at zero
and propagation only seeds negatives when `k_max_neg > 0`).

**Documentation sweep.** Reconciled the reference docs against the code: the `batch.py --backend` /
`--sam3-checkpoint` flags, the `merge_metric --neurons` / `--out-csv` / `--scale` flags, and the new
`eval.concat_merge_shards` and `eval.retro_eval` entry points are now in
[reference/cli.md](reference/cli.md); the `backend` / `sam3_checkpoint` config fields, the `kneg0`
presets, and the `k_max_neg = 0` semantics are in [reference/configuration.md](reference/configuration.md);
the new cluster and eval scripts are in [reference/code-map.md](reference/code-map.md); and the GUI's
`--em-scale` backdrop flag is in [how-to/review-flagged-chains.md](how-to/review-flagged-chains.md).
The submission plans (`run-sam3-on-narval.md`, `queued-sam3-narval-tests.md`) already carried the
run-time detail and are cross-linked from the reference pages.

<a id="r-2026-07-21-sam3-cluster"></a>
## 2026-07-21, SAM3 Phase 2: `--backend sam3` switch, cluster wiring, and the Narval runbook

The bake-off below (same day, read that entry first) left one open question: does SAM3's
2-chain win hold over the whole target-worm set. This round builds the plumbing to answer it,
without touching SAM2's default behavior anywhere. Spec
`docs/superpowers/specs/2026-07-21-sam3-cluster-whole-set-eval-design.md`, plan
`docs/superpowers/plans/2026-07-21-sam3-cluster-whole-set-eval.md`.

**What landed.** `batch.py` gained a `build_predictors(cfg)` helper and two `PipelineConfig`
fields, `backend: str = "sam2"` and `sam3_checkpoint: Optional[str] = None`, plus `--backend
{sam2,sam3}` and `--sam3-checkpoint PATH` on the CLI. `sam2` stays the default and is
byte-identical to before, the routing test asserts it, and a full-suite run confirms no prior
test changed. `cluster/run_array.sh` forwards `VENV`, `PRESET`, `SAM_BACKEND`, `SAM3_CKPT`, and
`OUT_ROOT` from the `sbatch --export` line through to `batch.py`, so a SAM3 run needs no file
edit, only different env vars on the submit line. A local single-chain `--backend sam3
batch.py` run confirmed the on-disk masks save and score through `eval.merge_metric` exactly
like SAM2's (the mask-format parity gate the plan called for before any cluster work), after one
small fix: `Sam3ImagePredictor.reset_predictor` was missing, and `batch.py`'s per-chain path
calls it between chains.

**The two whole-set runs, matched to the Phase-1 SAM2 baselines for a clean model swap:**

```
sbatch --export=ALL,VENV=$HOME/sam3env,PRESET=original_perslice_only_guard,SAM_BACKEND=sam3,SAM3_CKPT=<ckpt>,OUT_ROOT=<scratch>/target_perslice_only_guard_sam3 cluster/run_array.sh
sbatch --export=ALL,VENV=$HOME/sam3env,PRESET=original_tier2_s1forced_neg,SAM_BACKEND=sam3,SAM3_CKPT=<ckpt>,OUT_ROOT=<scratch>/target_tier2_s1forced_neg_sam3 cluster/run_array.sh
```

**The runbook.** `docs/how-to/run-sam3-on-narval.md` is the SAM3-specific delta on top of the
general `run-on-narval.md`, checkpoint upload, a fresh venv (`module load python`, `pip install
--no-index torch`, `pip install transformers>=5.13`, recorded once it works) with an Apptainer
container as the documented fallback built from the known-good local stack, smoke one chunk
(`--array=0-0`) before sizing the full array, then merge/download/score. `OUT_ROOT` is the shard
root, so each array task writes its own `chunk_<i>` shard exactly like the SAM2 baseline (no
shared-tree or manifest contention), and the runbook merges with `merge_shards.py --shard-root
$OUT_ROOT --out ..._merged` then downloads with `stage_download.sh` (which auto-globs
`*_merged`), the same flow the SAM2 runs use.

**Honest caveats, carried into the runbook.** SAM3 is roughly 3 to 4x slower per cell than SAM2
(the bake-off's own timing, so the smoke chunk gates the allocation ask), `pred_iou` is NaN for
SAM3 propagation (inert for the mask-only merge metric, but treat it as disabled for anything
else that reads it), and SAM3 must write to its own `OUT_ROOT`, never a SAM2 baseline tree: no
code enforces that beyond the path you choose.

**Not done here.** The actual whole-set numbers. That is the Narval run itself, human-executed
because Duo-MFA blocks headless login, tracked as the next step below.

---

<a id="r-2026-07-21-sam3"></a>
## 2026-07-21, SAM3 vs SAM2 bake-off: HF-transformers PVS adapters + 2x2 comparison

A postdoc shared a SAM 3 checkpoint (`F:\sam3\huggingface`, HuggingFace format) whose masks looked
better than ours. We tested it as a SAM 2 drop-in. Spec
`docs/superpowers/specs/2026-07-21-sam3-pvs-bakeoff-design.md`, plan
`docs/superpowers/plans/2026-07-21-sam3-pvs-bakeoff.md`, full API and numbers in
`docs/explanation/sam3-bakeoff-findings.md`.

**What landed.** Two thin adapters in `sam2_utils/sam3_backend.py` wrap SAM 3's Promptable Visual
Segmentation tracker (`Sam3TrackerModel`, `Sam3TrackerVideoModel`, via `transformers` 5.13.1, no new
install) behind the exact `SAM2ImagePredictor` and video-predictor surfaces the pipeline calls, so
`pipeline.segment_per_slice` and `pipeline.propagate` run SAM 3 unchanged. torch and transformers
import lazily, so the import-direction test stays green. A driver `experiments/sam3_bakeoff.py` runs
the 2x2 {SAM2, SAM3} x {propagation, per-slice} on shared chains, building one anchor seed and one
frame set per chain so only the model varies, and scores every cell with the same `eval.merge_metric`
and `sam2_utils.membrane` primitives. SAM 2 stays the pipeline default; `batch.py` is untouched.

**API findings (probe `experiments/sam3_probe.py`).** Both trackers fit the 6GB card (image 2.16 GB,
video 1.37 GB peak in bf16). Reverse propagation is supported (`propagate_in_video_iterator(...,
reverse=True)`), but SAM 3 does not auto-infer the start frame, so the video adapter records the
anchor frame and passes it as the default `start_frame_idx`. Two SAM3-vs-SAM2 divergences the adapter
absorbs: a box add needs `clear_old_inputs=True`, and bf16 mask logits need a `.float()` before
`.cpu().numpy()`.

**Results (target-worm merge metric, AIAL chain_05 = 17 frames, chain_00 = 113 frames).**
foreign-node rate orders the cells identically on both chains: sam2_prop worst, then sam2_perslice,
then sam3_prop, then sam3_perslice best. Long-chain foreign-node rate: sam2_prop 0.796, sam2_perslice
0.540, sam3_prop 0.372, sam3_perslice 0.221; dropout: sam2_prop 0.673 down to sam3_perslice 0.000.
Two effects stack: per-slice beats propagation (drift), and SAM 3 beats SAM 2 (bleed). **SAM 3
per-slice leads** (lowest bleed, zero dropout on both chains). Costs: SAM 3 is 3 to 4x slower per
cell, and its underfill runs higher than SAM 2 propagation's (the tight-mask cost, Phase 2c
grow-to-membrane is the lever). Caveat: two chains of one neuron on the GT-free merge metric, a strong
first signal, not a final verdict. Next step before productionizing (a `--backend sam3` flag) is a
broader multi-neuron run and ideally the cross-worm GT.

---

<a id="r-2026-07-21"></a>
## 2026-07-21, per-frame segmentation: two approaches, an AMG tuner, membrane-aware arbitration

A supervisor-requested complementary view to per-chain propagation: segment everything present in
one EM frame at once, and let the overlaps between cells do the work of resolving spills. Design:
`docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md`. New driver `run_perframe.py`;
no change to the existing per-chain batch/GUI path.

- **Why.** So far the pipeline segments one neuron chain at a time and propagates through z. This
  round builds the opposite view instead: segment every node-bearing cell in a single frame, then
  use the overlaps between cells to arbitrate spills. It deliberately delivers two roadmap items
  early, because this experiment gives them a concrete use: Phase-2 item 2d (a principled
  non-overlap resolve, in place of first-writer-wins) and a first, lightweight look at the R5
  dense-path hedge (segmenting a whole frame at once with SAM2's own auto-mask generator, rather
  than one prompted neuron at a time). Both are prototypes scoped to this experiment, not yet wired
  into the main per-chain composite; see [roadmap.md](explanation/roadmap.md).
- **Shared foundation.** A per-frame node index (`sam2_utils/perframe.py`'s `nodes_in_frame`,
  extending `merge_metric.nodes_by_z` to every cell at one z) and a per-frame scoring metric
  (`eval/perframe_score.py`'s `score_frame`), which reuses the Phase-2 membrane detectors
  (`spanning_membrane`, `boundary_on_membrane`, `underfill_fraction`) alongside own-node coverage,
  foreign-node bleed, and a pre-resolution pairwise-overlap scalar. Two overlap resolvers implement
  2d: `resolve_overlaps_argmax` (membrane-respecting nearest-node rule) and
  `resolve_overlaps_watershed` (seeded watershed on the membrane map), both pure array functions
  over masks, node coordinates, and the membrane map.
- **Approach 1, prompt-based (`--approach prompt`, the default).** Image-mode SAM2 once per node in
  the frame (positive point plus box, optionally the other cells' nodes as negatives), with a
  metric-guided multimask selector (`select_by_metric`) added alongside SAM2's own `pred_iou` and the
  existing `generous` policy. Three swept knobs, 12 combos total: `--negatives on|off`,
  `--selection pred_iou|generous|metric`, `--resolver argmax|watershed`. `--sweep` loops the whole
  grid over `--frames` and logs one row per combo.
- **Approach 2, auto-mask + match + keep-competitors (`--approach amg`).** Runs
  `SAM2AutomaticMaskGenerator` over the whole frame, matches each node to one of the resulting masks
  (`match_amg_to_nodes`, the F2 composite, or `--match area`, the smallest containing mask), and
  keeps the unmatched masks as unlabelled competitors that still take part in overlap resolution
  before being dropped to background. This is what lets a neighbour push bleed off a cell mask
  without ever appearing as a named cell itself.
- **AMG parameter tuner (`--tune`).** Grid-search over `pred_iou_thresh` x `stability_score_thresh`
  x `points_per_side` (12 combos by default, `--tune-grid` overrides), scored by
  `eval.perframe_score.objective`, a composite that rewards own-node coverage and
  boundary-on-membrane while penalising foreign bleed, spanning, and overlap. Every trial's params
  and per-frame scores land in `<out>/trials.csv`; the winning trial gets a full re-run for its
  montages, and the logged summary carries an explicit note that the objective can be gamed (a
  degenerate small-mask trial can score well on paper while under-covering the frame).
- **Documentation, a first-class requirement of the design.** Every run writes
  `results/perframe/<run>/{config.json,scores.csv,montages/}` (gitignored, regenerable); the
  committed [perframe-experiments.md](explanation/perframe-experiments.md) logs one row per run plus
  a "Comparison protocol" section describing how Approach 1's best sweep knobs, Approach 2 default,
  and Approach 2 tuned are meant to be judged against each other on the same frame sample.
- **Built as two plans, six-plus-three tasks, via subagents.** Plan 1 (foundation, F1 to F3, the
  Approach-1 runner, the sweep) and Plan 2 (the Approach-2 runner, the tuner, this documentation
  pass). 257 tests pass, ruff clean.
- **The comparison is a documented protocol, not yet a finished result.** Approach 2's default AMG
  parameters (`points_per_side=64`, `crop_n_layers=1`) proved too slow to finish even a single
  target-worm frame in a short local wall-clock budget, and a `--tune` run multiplies that cost by
  the grid size. The full three-way comparison on a 5-to-10-frame sample is realistically a CCDB
  job; locally we have a single-frame worked example (frame 1400) showing how to read the numbers
  against the montages once the full sample lands. See "Comparison protocol" in the experiments log
  for the concrete caveats this surfaced, including that `--match area` can let one merged blob wear
  two cells' names.
- **Fixed: fair own_coverage for Approach 2, then a unified pre/post scoring contract for both.**
  Two follow-up fixes to the metric above, landed the same day. First, Approach 2's `own_coverage`
  used to score only the cells an AMG mask actually matched, so an unmatched node was simply absent
  from the mean instead of counting as a dropout; unmatched cells now get an empty resolved mask and
  count as uncovered, matching Approach 1. Second, and more fundamentally, Approach 1 was scoring its
  raw pre-resolution union masks (so `--resolver` never moved a single number) while Approach 2 was
  already scoring its resolved, post-`--resolver` masks (so its `overlap_fraction` read ~0 by
  construction); the two approaches now share one contract, `own_coverage`/`total_foreign`/
  `mean_boundary_on_membrane`/`spanning_rate`/`mean_underfill` are always scored on the RESOLVED
  masks for both, and `overlap_fraction` is deliberately overridden to a pre-resolution diagnostic
  (the raw masks' pairwise pixel fight before `--resolver` sorts it out) computed the same way for
  both approaches. This makes `--resolver` an actual knob for Approach 1 and makes the two
  approaches' numbers comparable. See [perframe-experiments.md](explanation/perframe-experiments.md)
  for the updated worked example and the note that its older table rows predate this fix.

---

<a id="r-2026-07-20"></a>
## 2026-07-20, Phase 1 close-out: blow-up guard + generous-first/negatives-in-crop bundle

Two more roadmap Phase 1 fixes, both gated off by default, plus the three presets that measure them.
Design: `docs/superpowers/specs/2026-07-17-phase1-blowup-guard-and-genfirst-negcrop-design.md`. No
default pipeline behavior changed.

- **Why.** The Phase 2 retro-score (previous entry) settled that per-slice re-seeding's real cost is a
  gross tail, a handful of slices blowing up to the whole worm cross-section, not mild bleed. Separately,
  the 2026-07-15 experiment found negatives cut bleed in the crop, but the existing tier-2 rerun always
  inherits the first pass's seeds, so there was no way to run a generous, negative-free first pass (so
  the crop is not clipped) and then turn negatives on only in the crop.
- **Per-slice blow-up guard (`pipeline/propagate.py`, `PipelineConfig.blowup_guard`).** A post-pass at
  the end of `segment_per_slice`, run only when `per_slice_reseed` is on: compute the median area over
  the chain's non-empty masks, treat any mask over `blowup_area_factor` (default 25.0) times that
  median as a blow-up, and replace it with the nearest accepted slice's mask by frame-index distance.
  Guarded frames get `pred_iou` set to 0.0 (the QC confidence signal that queues them for review;
  `frame_conf` is zeroed too, for consistency) since a substituted neighbour mask is a stand-in, not a
  real segmentation of that slice.
  A chain with too few accepted masks or a zero median skips the guard rather than picking a spurious
  baseline. Off by default and inert on the video-propagate path, so existing runs are byte-identical.
- **Per-pass tier-2 seed overrides (`pipeline/config.py`, `batch._run_one_chain`).** Three new
  `PipelineConfig` fields, `tier2_k_max_neg`, `tier2_seed_negatives`, `tier2_multimask_generous`, each
  `None` by default (inherit the base value, current behaviour unchanged). When set, the tier-2 rerun
  applies them on top of `chain_crop=True`, so the first `_sam` pass and the tier-2 crop pass can seed
  differently, generous and negative-free to size the crop, then negatives on and generosity off once
  inside it.
- **Three presets (`sam2_utils/presets.py`).** `original_perslice_only_guard` and
  `original_perslice_guard` add the blow-up guard to the existing `original_perslice_only` /
  `original_perslice` trees, isolating the guard's effect against their guard-off counterparts.
  `original_genfirst_negcrop` runs the generous-first, negatives-in-crop bundle as a `tier2_all`
  two-pass: a generous, negative-free first pass sizes the crop via `chain_crop_from_mask`, then the
  tier-2 overrides turn negatives on for the crop pass.
- **Built as three TDD tasks via subagents** (the guard, the seed overrides, the presets), then this
  documentation pass. Tests and ruff stayed green throughout.
- **CCDB A/B verdict (2026-07-21): per-slice + guard graduates, the genfirst bundle is dropped.** The
  three presets ran on Narval and were scored on `eval.merge_metric` (membrane included) over the
  629-chain subset. The blow-up guard cut per-slice's gross tail (`total_foreign`) by 73%: 17,481 to
  4,776 for `perslice_only`, 28,769 to 7,702 for `perslice`, with dropout still near 0. `perslice_only
  + guard` (no generous) beats the `tier2_s1forced_neg` baseline on foreign-frame-rate (0.109 vs 0.321),
  dropout (0.001 vs 0.130), and mild-bleed (0.016 vs 0.029); its residual tail (4,776) sits near the
  baseline's 3,570, and its only weak column is underfill (0.616 vs 0.483, the cost of tight masks).
  Generous still hurts (`perslice_only + guard` beats `perslice + guard`, 0.109 / 4,776 vs 0.182 /
  7,702). `genfirst_negcrop` ties the baseline (foreign 0.328, dropout 0.114, total_foreign 3,583) at
  about 2.5x the compute, so it is rejected. Phase-1 exit decision: per-slice re-seeding plus the
  blow-up guard, without generous, is the leading candidate; the residual underfill points to
  grow-to-membrane (Phase-2 item 2c) as the next lever.

---

<a id="r-2026-07-17"></a>
## 2026-07-17, Phase 2 foundation: membrane map + membrane-aware bleed detection

The roadmap Phase 2 foundation (2a + 2b), landed. Design:
`docs/superpowers/specs/2026-07-17-phase2-membrane-map-bleed-detection-design.md`; decision record:
[ADR 0016](adr/0016-membrane-map-border-to-border-bleed-detection.md). No default pipeline behavior
changed, this is a new measurement, not a new lever.

- **Why.** The Phase-0 skeleton merge-metric (ADR 0015) is a severe-merge floor: it only fires once a
  mask reaches a neighbour's centreline, so it is blind to mild bleed (a mask crosses a real membrane
  but stops short of the neighbour's node) and to underfill (a mask stops short of its own cell's
  membrane). Neither the QC flags nor the Phase-0 metric had any opinion on either.
- **2a, the membrane map (`sam2_utils/membrane.py`).** `membrane_map` reads a grayscale EM patch and
  returns a per-pixel membrane-ness map in `[0, 1]`, v1 a Sato dark-ridge filter normalised by its own
  99th percentile. The signature is a swappable interface, a trained model can sit behind it later
  without touching anything downstream.
- **2b, the detectors (same module).** Three pure array functions, mask and membrane map in, scalars
  out: `spanning_membrane` (does a membrane ridge cut the mask border-to-border, the mild-bleed
  signal), `boundary_on_membrane` (fraction of the mask perimeter sitting on a membrane, a
  direction-agnostic boundary-quality check), `underfill_fraction` (a bounded outward flood measuring
  how much room the mask left before its enclosing membrane). The border-to-border criterion in
  `spanning_membrane` is deliberately not "any membrane pixel inside the mask": a nucleus is a closed
  interior loop, not a ridge that spans the mask, so a soma is never falsely flagged, with no
  special-cased nucleus detection anywhere in the code.
- **Wired into the scorer (`eval/merge_metric.py`).** A new `MembraneSource` loads and caches the raw
  EM per z through the existing FrameStore seam and crops it to each mask's window, so the membrane
  pass degrades gracefully (falls back to Phase-0-only) whenever the EM for a frame is unavailable.
  `score_chain` now carries four columns per frame (`spanning_merge`, `bled_fraction`,
  `boundary_on_membrane`, `underfill_fraction`), and `score_run`'s summary gains
  `mild_bleed_rate` (the headline: a spanning membrane crossing with no foreign node, the mild bleed
  the Phase-0 floor cannot see), `spanning_merge_rate`, `mean_boundary_on_membrane`, and
  `mean_underfill_fraction`. New CLI flags: `--no-membrane`, `--tau`, `--tol`. Full reference:
  [cli.md](reference/cli.md), [configuration.md](reference/configuration.md).
- **Built as five TDD tasks via subagents**, the membrane map, the three detectors, the
  `MembraneSource` loader, the scorer wiring, then this documentation pass. 232 tests pass, ruff
  clean.
- **Deferred on purpose, each to its own spec:** 2c, grow-to-membrane refinement, which would reuse
  the `underfill_fraction` flood to actually change a mask rather than only measure the gap; and 2d,
  replacing the composite's first-writer-wins with a membrane-aware non-overlap resolve. Both need
  this ruler in place first to be gated fairly, which is exactly why this round stopped at measurement.
- **Follow-on: `--scale` override + a one-frame visual sanity check + the retro-score.** Added a
  `--scale` flag so trees without a `_run_meta.json` (the merged CCDB shards) can be scored, and saved
  two example membrane-map figures under `docs/figures/membrane-v1/`. The visual check confirmed the
  v1 map traces real cell boundaries but also fires on organelles (mitochondria, vesicles), so it is a
  comparative ruler, not an absolute one, and on a soma the nucleus envelope is a stronger ridge than
  the faint outer membrane (the nested-membrane ceiling seen from the membrane side).
- **Retro-score verdict on the four Phase-1 A/B trees** (scale 8; Phase-0 numbers reproduced the
  2026-07-15 run exactly, confirming the scale). Membrane columns, baseline / generous_only /
  perslice_only / perslice: mild_bleed 0.029 / 0.028 / 0.020 / 0.019; spanning_rate 0.095 / 0.111 /
  0.034 / 0.061; underfill 0.483 / 0.469 / 0.615 / 0.475; boundary_on_membrane ~0.90 flat. Reading:
  (1) **per-slice's damage is gross blow-ups, not mild bleed.** Its typical mask is the cleanest of the
  four (lowest mild_bleed and spanning_rate); all its cost sits in the gross tail (`total_foreign`
  17,481, a whole-worm blob is not "spanning" so it lands in Phase-0, not mild_bleed). This overturns
  the earlier "trades severe for mild bleed" guess: the fix is still the blow-up guard. (2) **per-slice's
  real tradeoff is underfill** (0.615, highest), the honest cost of tighter masks. (3) **generous shows
  its intended benefit and its cost, measurably:** underfill down (0.483 to 0.469) while spanning_rate
  up (0.095 to 0.111), a genuine fill-vs-merge lever rather than a pure loss. (4) `boundary_on_membrane`
  is not discriminative at v1 (organelle noise). Caveat: read mild_bleed alongside Phase-0, never
  instead of it; the two partition gross vs subtle merge.

<a id="r-2026-07-15"></a>
## 2026-07-15, negatives experiment + measurement-first roadmap redesign

A follow-up experiment round and a roadmap restructure. No default pipeline behavior changed.

- **Negatives + full-res second-pass experiment.** Two presets on the two-step baseline:
  `original_tier2forced_neg` (crop_scale 2 + `seed_negatives`) and `original_tier2_s1forced_neg`
  (crop_scale 1 + negatives), run on Narval over EXP_NEURONS. tier2forced_neg stopped early (AIBR
  unfinished); tier2_s1forced_neg completed. Pure-flag result on the 582-chain common set: adding
  negatives or the full-res second pass is near flag-neutral (flagged chains 94, 99, 103; frame rate
  0.205, 0.202, 0.192; negatives net -5 flagged chains). Load-bearing caveat: the QC flag rule has no
  bleed signal, so these counts cannot grade what negatives target. A real verdict needs the
  merge-metric below.
- **Third deep-research pass (prior art), adversarially verified.** Confirmed the direction and
  corrected two beliefs: the connectomics-standard rulers are skeleton/topology metrics (ERL, VOI),
  which our CATMAID skeletons already support, and the membrane-map + skeleton-expansion filling is the
  lab's own prior method (Mulcahy/Witvliet), the direct precedent for a per-frame boundary map.
- **GT erosion confirmed for our copy.** The cross-worm VAST segmentation is unpublished/incomplete and
  its neighbouring masks are inset from the shared membrane by design, so boundary metrics against it
  are biased and the different-worm mismatch compounds it. This does not match the published C. elegans
  filling (which expands to the membrane), so it is a property of our copy, not the method.
- **Roadmap redesigned (`roadmap.md` §5) into a measurement-first, evidence-gated Phase 0-4 plan.**
  Phase 0 fixes the ruler with a GT-free target-worm merge-metric (foreign skeleton-node containment on
  raw masks) and retro-scores the existing runs; Phase 1 is cheap structural fixes (per-slice re-seed,
  mutex-watershed non-overlap, prompt fixes for the new nested-membrane ceiling); Phase 2 is the
  per-frame membrane map; Phase 3 is the boundary benchmark plus mask-decoder finetune / FGNet head;
  Phase 4 is the dense-3D paradigm gate. Added problem #7 to §2 (the point-prompt ceiling on
  double-bordered somas). §4 stays as the solutions-by-problem reference.
- **Phase 0 merge-metric built and run (`eval/merge_metric.py`).** A ground-truth-free severe-bleed /
  dropout scorer: for each RAW per-chain mask it counts foreign skeleton nodes contained (a merge) and
  own-node dropout, scored against the target worm's own CATMAID skeletons (no cross-worm GT, no
  boundary dependence). Retro-scored the five runs:

  | run | foreign_frame_rate | dropout_rate | total_foreign |
  |---|---|---|---|
  | fullres (neg off) | 0.356 | 0.359 | 3941 |
  | wholeimg_s4 (neg off) | 0.453 | 0.319 | 5749 |
  | tier2forced (neg off) | 0.471 | 0.143 | 6685 |
  | tier2forced_neg | 0.357 | 0.147 | 3725 |
  | tier2_s1forced_neg | 0.321 | 0.130 | 3570 |

  The verdict the flag tables could not give: cropping cuts dropout about 2.5x (the mask stays on its
  own cell), neg-off cropping has the highest bleed (a tight crop fills aggressively), negatives then
  cut that bleed (foreign_frame_rate 0.471 -> 0.357, foreign nodes 6685 -> 3725) at no dropout cost,
  and full-res crop + negatives (tier2_s1forced_neg) is best on every axis (bleed 0.321, dropout
  0.130). So negatives do help; the flags were simply blind to bleed. Sanity check: AVAL/ch16 (the
  known wrong-cell jump) shows 111 foreign-hit frames in fullres vs 13 in tier2_s1forced_neg. Caveats:
  this is a severe-merge floor (foreign-node containment, radius 3), it does not see mild bleed (that
  needs the Phase 2 membrane map); and tier2forced_neg covers 582 chains (AIBR unfinished), so its
  rates compare but its totals do not.
- **Phase 1 code landed (gated off, awaiting GPU smoke).** Two roadmap Phase 1 fixes, both behind
  default-off flags so existing behavior is byte-identical: per-slice re-seeding (`per_slice_reseed`,
  run_chain segments each slice from its own skeleton node in the chain crop with no video
  propagation, so memory cannot carry the wrong cell across slices) and generous-capped multimask
  (`multimask_generous`, prefer a larger candidate so a soma includes the nucleus and reaches the
  outer membrane, capped so a whole-frame blob never wins while a gate-passer exists). New
  `original_perslice` preset. Built as 8 TDD tasks via subagents; the whole-branch review caught and
  fixed a frame_conf low-res-logit indexing crash before it could reach a GPU run. 215 tests pass,
  ruff clean. The real quality A/B (per-slice and generous vs tier2_s1forced_neg on eval.merge_metric,
  foreign_frame_rate down, dropout not up) is pending the human GPU smoke: downscaled locally, then
  CCDB. Spec + plans under docs/superpowers/.

<a id="r-2026-07"></a>
## 2026-07, research passes, resolution experiments, and design specs

A research and experiment round on top of the M4.5 GT/eval work. No default pipeline behavior
changed; this was measurement, cluster experiments, and design.

- **Two verified deep-research passes**, both adversarially fact-checked and folded into the
  roadmap (§4, §5). Error detection plus benchmark design moved Stage 4's cheap training-free
  pieces forward and added the target-worm annotation-benchmark requirement. Segmentation
  improvement produced the neurite-targeted-finetune hard rule (an organelle-trained EM model
  degrades neurites) and confirmed the dense/hybrid path is a learned, evidence-gated last
  resort, not a drop-in.
- **Resolution experiments on Narval.** Five target-worm variants added as presets and run as
  Slurm arrays (`cluster/run_exp.sh` + a per-variant merge + `cluster/stage_download.sh`):
  `original_fullres` (whole image at scale 1, no tier-2), `original_wholeimg_s4` (a scale-4
  control), `original_tier2forced` (tier-2 on every chain, fallback floor dropped),
  `original_bigimg` (SAM2 `image_size` raised to 2048), and `original_tier2_s1` (scale-1 tier-2,
  built). Findings: (1) `bigimg` crashed on every chain with a SAM2-internal `.view()`/stride
  error at `image_size` other than 1024, off-distribution for the pretrained weights and not our
  code, so it is retired as configured; (2) the effective resolution of a crop is
  `min(crop_tif/crop_scale, 1024) / crop_tif`, so the default `crop_scale=2` feeds the median
  crop (1560 tif) at about 780 input px, under-filling SAM2's 1024 input for 86% of chains, and
  `original_tier2_s1` fills it for roughly a third more resolution at no cost; (3) whole-image
  scale 1 is not high resolution, SAM2 downsamples it to 1024, so the large saved files are a
  high-resolution rendering of a low-resolution segmentation.
- **Design specs (not yet implemented)** under `docs/superpowers/specs/2026-07-07-*`: a GUI
  run-picker that opens runs by their `_run_meta.json` metadata (and fixes re-segmentation scale
  and hires-EM); mask-seeded propagation (seed the video from the anchor mask, gated on anchor
  quality, with a saved-mask reuse hook); and coarse-to-fine tiled full-res propagation (reuse a
  cached coarse pass, tile over its foreground, mask-seed each tile, union). A branch-junction
  multi-seed idea is in discussion, gated on measuring whether the per-neuron union under-covers
  junctions.

<a id="r-2026-06"></a>
## 2026-06, review tooling + post-processing pass

A round of review-GUI and mask-cleanup work on top of the M4 review tool.

- **Per-chain GUI (`gui.py`) upgrades.** The single queue picker became a mode toggle
  (flagged-only / everything) plus cascading neuron then chain selectors, so any chain on
  disk is openable, not just flagged ones (`ReviewQueue.all_chains` / `chain_status`).
  Added image-phase box prompts (a drawn box seeds the `R` re-predict alongside points;
  `image_predict` now forwards `box_sam`). Added tier-2 recrop: grow the crop by N tif px
  (`C`) or draw a re-centered window on the full frame (`F`), both re-running the chain via
  a new `run_chain` `override_crop_window`. Added save-masks-now (`S`) to persist the mask
  layer without re-propagating. Dropped the per-frame mark-ok/wrong buttons (keys stay) and
  wrapped the dock in a scroll area so nothing is buried.
- **Tier-2 crop sizing.** `chain_crop_pad_tif` default raised 64 -> 512 (windows that
  looked fine clipped the cell in practice). Added a collapse fallback: when the first pass
  left no usable mask, size a fixed `chain_crop_collapse_size_tif` (1024) window on the
  anchor node (`node_crop_window`) instead of a skeleton-only guess. See
  [0009](adr/0009-tier2-crop-fallback.md).
- **Neuron-level review GUI (`gui_neuron.py`), a second paradigm.** Opens a whole neuron on
  one per-neuron crop (`_ncrop`); branches stay separate SAM2 objects shown as one
  multi-color Labels layer; corrections act on the active branch over a z-scoped view; the
  skeleton and the branch's saved seed are shown; neuron-level approve/reject; overlay
  export to mp4 / png / gif. New library support: `neuron_crop_window`,
  `remap_mask_to_window`, `video_viz.to_png_seq`. See
  [0014](adr/0014-neuron-level-review-gui.md) and the 2026-06-23 spec/plan under
  `docs/superpowers/`.
- **Size-aware mask post-processing.** `remove_small_islands`, `fill_small_holes`,
  `smooth_edges` in `masks.py`, wired under the `postprocess_masks` master toggle with new
  per-op knobs (off by default), and `batch.py --postprocess` / `--no-postprocess` for an
  A/B run. See [configuration.md](reference/configuration.md).

---

<a id="old-2"></a>
## old §2, Milestone-by-milestone build narrative

> Moved from design-notes.md §2 "Where we are now". This is the running build log
> from M1 through the M4 review-testing pass. The live doc now carries a short current-state
> summary instead; this is the full narrative.


**Milestones 1 and 2 are complete** (see §6). The notebook has been lifted into
`pipeline.py`, phase functions plus a `run_chain` driver and `ChainState`
serialization, driven by a thin `run_aval.py` bootstrap. `run_chain` reproduces
the notebook's AVAL masks pixel-for-pixel (verified by diff on the AVAL chain),
with `ChainState` persisted to `state.json`. The notebook below remains the
reference for *what* each phase does; `pipeline.py` is now the source of truth for
*how* it runs.

**M2 (inline QC + flagging) landed.** `run_chain` now has a 9th phase, `run_qc`,
that runs `qc.compute_metrics` over the just-saved chain, writes `qc.csv`,
populates `ChainState.qc_summary` / `triage_frames`, and sets the chain's
`status` to `done` / `flagged`, all headless. QC thresholds are exposed as
`qc_*` knobs on `PipelineConfig` (one place to tune; see §7). The first AVAL run
flags ~38% of frames with 5 `intervene` (≥2-signal) frames, a sane starting
point pending threshold tuning. See §5 for the bugs this surfaced.

**Frame-prep reuse landed (M3 support).** `prepare_video_frames` no longer
re-decodes per chain. It keeps a shared decode cache at
`frames_root/frames_cache_s{scale}/z{file_z}.jpg` (each EM frame downscaled once
ever) and builds a per-chain 0-indexed *link view* at
`frames_root/chain_views/{neuron}_chain{idx:02d}_s{scale}/` (symlink → hard-link
→ copy fallback; on Windows the hard-link branch is the usual path, and works
because cache and views share one volume). Overlapping chains now pay the ~9k×9k
imread+resize once across the dataset instead of once per chain, the prep
bottleneck. Views are namespaced by neuron+chain so a multi-neuron batch can't
collide, and are rebuilt fresh each call. Cached JPEG bytes are byte-identical to
the old per-range writer, so AVAL still reproduces pixel-for-pixel; `run_chain`'s
external signature is unchanged. (Old `sam2_video_{start}_{end}_s{scale}/` folders
from prior runs are now orphaned, safe to delete.)

**M3 batch runner, in place, with reset / scope / telemetry knobs.** `batch.py` is
the headless driver: build the session once, enumerate chains from `chains.json`, run
each through `run_chain`, record status to `output/_manifest.csv` (crash-safe atomic
rewrite per chain; a `running` breadcrumb so an interrupted chain is retried next
launch), and roll per-chain flags up into `output/_triage.csv`. Resume policy is in
`_should_run` (done/flagged skip; failed retries by default; pending/running run). Two
run knobs are surfaced on `run_batch` / `main`: **`neurons`**, an allow-list (e.g.
`["AVAL", "AVAR"]`) scoping a partial run, `None` = all; and **`clean`**, wipe prior
outputs and start fresh. `clean` is scope-aware: with `neurons=None` it removes the
whole `output_root` (full reset); with a subset it deletes only those neurons' chain
dirs and prunes their rows from `_manifest.csv` / `_timing.csv`, leaving other neurons'
finished work intact. `clean` differs from `force`: `force` re-runs in place and lets
`save_masks` overwrite (but `save_masks` never clears `masks/`, so a chain whose frame
coverage shrank leaves orphan PNGs that QC then re-scores), `clean` deletes first, so
QC only ever scores the current run. Per-chain overlay gifs via `gif_mode`
(`off` / `flagged` / `all`). Per-chain timing + peak VRAM are written to
`output/_timing.csv` (see §7, runtime telemetry).

**Repo hygiene pass (June 2026, pre-GUI).** A lean-up before M4. Coordinate
transforms were fully centralized into `alignment.py` (see §4). Dead code removed:
`sam2_utils/diag_utils.py` (superseded by `diagnostics.py`) and the orphan
`single_object_depth_segmentation.py` script (superseded by `pipeline.py`; the
`_.ipynb` reference notebook stays). Data + output paths now have one home in
`sam2_utils/config.py` (`CSV_PATH` / `CHAINS_PATH` / `ROOTS_PATH` / `OUTPUT_ROOT` /
`FRAMES_ROOT`); `run_aval.py` and `batch.py` import them instead of re-declaring
absolute paths. `diagnostics.py` no longer imports torch at module top (lazy), so
`import sam2_utils` works on a torch-free box. `ChainState.phase_seconds` /
`phase_subseconds` are now declared fields and serialize into `state.json` (were
stamped-on attributes, lost on resume). `batch.py`'s `gif_mode` now honors
`off`/`flagged`/`all` correctly and writes under the passed `output_root`. A
torch-free test suite (`tests/test_alignment.py`, 13 cases) guards the transform
math: run `py -3 tests/test_alignment.py` (or pytest). The only behavioural change
in all of this is the new loud `run_qc` guard; everything else is structure, so a
fresh `batch.py` run reproduces as before.

**M4 (napari review/triage GUI), core landed (June 2026).** The fourth thin
driver, `gui.py`, plus two torch-free helpers it owns: `sam2_utils/review_queue.py`
(the work queue + a GUI-owned `_review.csv` disposition ledger, kept separate from
the batch's `_manifest.csv`) and `sam2_utils/labels.py` (the per-frame **label
engine**, `_labels.csv`, the M4.5 training data). The GUI reads the batch's flagged
chains, lets a human scrub to flagged frames, edit positive/negative prompt points,
paint an anchor mask, re-run the image phase, and resume propagation over
`PropagationSession`, then rewrites `masks/` + `qc.csv` + `state.json` so a corrected
chain is byte-indistinguishable on disk from a fresh batch run. GPU is lazy (browse/
label needs no predictors). The interactive `PromptRefiner` / `PointClicker`
matplotlib-widget prototypes (below) are now **superseded** by this, they were the
sketch; `gui.py` is the napari rebuild they called for. What's left for M4.5 is
*training* on the labels M4 collects, plus the label-gated accuracy levers; see §6
row 4 for the shipped-vs-deferred split and §7 for the deferred items.

**M4 review-testing pass (June 2026).** Fixes from first real use of the GUI:
(1) **Directional resume**, a correction now re-propagates *away from the anchor only*
(anchor → both ways; a frame after/before the anchor → forward/reverse only), so an
already-corrected center frame is never clobbered when you fix a later one. (2) **Mask
seed, box dropped**, GUI corrections seed propagation with the mask (`add_mask`: the
re-predicted and/or hand-painted mask on the frame), never a derived box; this is the
§7 *box-vs-mask* "human-painted mask is the maximally-verified seed" path, now the GUI
default (the box overlay was removed). (3) **Queue cycling**, prev/next CHAIN cycle
through every undisposed chain *including `in_review`* and wrap, fixing "can't return to
an unfinished chain" (opening a chain marks it `in_review`, which the old next-button
excluded → false "queue empty"). (4) **Painted corrections persist across resumes**, the GUI video predictor is now built with SAM2's `add_all_frames_to_correct_as_cond=True`
(`setup.build_predictor(kind="video", correct_as_cond=True)`). Without it, painting/clicking
a frame the session had *already tracked* (the normal paint→resume→inspect→repaint loop)
demoted the correction to a *non-conditioning* frame, so the next `propagate_in_video`
re-inferred that frame from memory with `mask_inputs=None` and **silently reverted the
paint** to its pre-correction state. (The *first* correction on a freshly opened chain was
unaffected, an untracked frame is an *initial* conditioning frame regardless of the flag, which is why it surfaced only on iterative re-touches.) The flag is the SAM2-documented
mechanism ("a frame that receives a correction click becomes a conditioning frame"; it's
`True` in Meta's MOSE-finetune config) and is **inert for the headless batch**, `propagate`
only ever seeds the anchor as an *initial* conditioning frame and never corrects a tracked
frame, so the M1 AVAL pixel-for-pixel reproduction is unchanged and the flag stays
default-off there. This makes the §7 *box-vs-mask* "human-painted mask is the
maximally-verified seed" guarantee actually hold across a multi-correction session. Three
larger items were **documented, not built** (at
the lab's discretion, before/around M4.5): a **marking/intervention GUI split** (sweep
ok/bad in a marking mode, fix only flagged frames in an intervention mode, the
too-many-buttons + scroll-confusion fix), **strict-by-default flagging** (flag
aggressively now for recall, loosen once the M4.5 detector can set the operating point),
and **higher-res masks** (the "pixel-art" resolution complaint, the real fix is the
M4.5 tier-2 per-chain crop; `--hires-em` is the interim EM-only sharpening). See §7.

The notebook `single_object_depth_segmentation_.ipynb` does one chain
end-to-end:

1. Pull CATMAID annotations, apply the stack→tif affine (`alignment.catmaid_to_tif`).
2. Decompose a neuron into MLC chains; pick one chain, take its **mid frame** as anchor.
3. **Image mode** on the anchor: positive skeleton node + K nearest neighbors as
   negatives → mask → largest connected component → bounding box.
4. **Video mode**: seed the box+point on the anchor frame, propagate **bidirectionally**.
5. Save per-frame PNGs.

Reusable pieces already in `sam2_utils/`:

- `config` / `setup` / `catmaid` / `alignment`, stable, keep as-is.
- `viz`, `video_viz`, static + animation display (notebook-oriented).
- **`qc`**, computes the failure signals (see §5) and the composite flag rule;
  thresholds are now parameters. Wired into `run_chain` via `pipeline.run_qc`
  (M2). This was the half-built core of the auto-detection milestone; it is now
  load-bearing.
- **`review`**, read-only proofreading viewer (added alongside M2). Rebuilds the
  overlay from a finished chain's on-disk artifacts (`masks/`, `state.json`,
  `qc.csv`) and delegates rendering to `video_viz`; `grid_flagged` / `animate_flagged`
  show only the QC-flagged frames. Strictly read-only by design, it is NOT the
  M4 intervention GUI (one correction tool, §4). Reuses `qc._iter_mask_paths` /
  `qc._load_binary` so mask reading has a single definition.
- `diagnostics`, VRAM/RAM/disk snapshots for long GPU runs.

The interactive `PromptRefiner` / `PointClicker` classes in the notebook are
matplotlib-widget based and currently **unresponsive in Jupyter**. They are the
prototype for the refinement GUI but should be rebuilt in napari, not patched.

**M4.5 measurement round landed (June 2026), see §8 for the full A/B + decisions log.**
Tier-2 per-chain cropping (`chain_crop`, default-off) is now safe and fast: an `image_score`-gated
fall-back to `_sam` when a crop anchor is poor (item b), and a windowed memmap frame read that is
bit-identical and ~48× faster (item c). The video anchor seed is now a sweepable spec
(`seed_box`/`seed_points`/`seed_negatives`/`seed_mask` + `box_margin_frac`), defaults reproducing
M1. Measured outcomes: **box+positive is the best AUTO seed** (mask-seed needs a high-quality anchor
→ GUI/tier-2 only; negatives are chain-dependent → default-off); **`box_margin_frac` fixes real
under-filled anchors** (validated, kept as a targeted lever); **tier-2+fallback is regression-free**
across 3 neurons (net queue −10). Commits `0155c2b`, `8f00330`, `2cf448e`. **Both lab decisions now
resolved (June 2026):** (1) **tier-2 default-on for flagged chains, DONE**, landed as an automatic
batch second-pass (`batch.py` `tier2_on_flagged`, default on; flagged `_sam` chain re-runs once with
`chain_crop=True`, regression-free via the item-b fallback), see §8.8 + `tests/test_tier2_rerun.py`;
(2) **(e) re-propagate-corrected-`_sam`-as-tier-2 GUI path, DEFERRED but kept LIVE** (not dropped):
§8.8 makes flagged chains crisp-paintable before review (§8.6 interaction note), so (e) now serves
only the narrow `_sam`-in-GUI set (fell-back / in-GUI-flagged), revisit if the fallback rate is high.
See §8.7 for the build sketch.

---

---

<a id="old-5"></a>
## old §5, Known issues: full resolution stories

> Moved from design-notes.md §5 "Known issues to resolve in the refactor". The live doc
> now carries a trimmed "Invariants & gotchas" version of §5 (the catalogue your tooling
> points at); this is the full text including the detailed how-it-was-fixed narratives for
> issues now closed.


These currently break or will break the QC step, fix them when extracting the library.

**Mask-space / filename issues (§5.1, §5.2), RESOLVED in milestone 1.**
Canonical on-disk mask space is now fixed in one place (`PipelineConfig`):
masks are stored at `_sam` space with `save_downscale == scale == 8`, so there is
no resample and no 2× skeleton-containment offset. Files are named
`mask_<catmaid_z:04d>.png` (no `z` prefix), which `qc._iter_mask_paths` parses,
and are written as **0/255 uint8 single-channel** (the notebook's format) by
`pipeline.save_masks`, directly viewable and pixel-comparable to the notebook.
`qc._load_binary` thresholds `> 0`, so it reads them unchanged. Note: this means
`pipeline.save_masks` does **not** use `qc.save_masks`, which writes uint16
*instance labels* (foreground pixel == obj_id). For a single object obj_id is 1,
and value-1 in a 16-bit image looks empty and is destroyed by any 16→8-bit
conversion, that was the original "empty masks" red herring. Instance-label
encoding is a multi-object concern, deferred to M5 (see §7).

1. ~~**Filename convention mismatch.**~~ Resolved, see above. (`mask_<catmaid_z:04d>.png`.)
2. ~~**Mask-space mismatch.**~~ Resolved, see above. (`save_downscale == scale`, no resample.)
3. **`pred_iou`, RESOLVED (populated, June 2026).** SAM2 *computes* the mask-decoder
   IoU head (`ious`) but `track_step` discards it before it reaches `inference_state` or
   the `propagate_in_video` yield (trace: `_forward_sam_heads` → `track_step` in
   `sam2/modeling/sam2_base.py`; the value is unpacked to `_`). It is still in hand one
   level down, in `_track_step`'s return (`sam_outputs[2]`). `pipeline._attach_iou_hook`
   wraps `_track_step` **read-only** to record `float(ious.max())` per frame (max == the
   argmax SAM2 itself selects on a multimask anchor; the lone value when single-mask).
   `propagate()` now returns `(video_segments, frame_conf, pred_iou)`; `run_chain` maps
   frame_idx→z and hands it to `run_qc`, which forwards it to `qc.compute_metrics(pred_iou=…)`.
   The hook is **best-effort**: if a future SAM2 refactor changes `_track_step`, pred_iou
   falls back to NaN (the flag rule already treats NaN as inert) rather than crashing.
   **Consequence:** the 4th flag signal is now live, `flag_count` can reach 4, and the
   "stable-but-wrong" frames the §7 item-4 *Bound* called out (plausible area, good
   temporal overlap, node inside mask) can finally trip a signal. `frame_conf`/`logit_conf`
   (per-frame mean-foreground-sigmoid confidence, recorded in `qc.csv`) stays as a secondary
   diagnostic, no longer the only confidence proxy. *(Earlier state, for the record: at M2
   `propagate` already stopped throwing the logits away and returned `frame_conf`, but the
   calibrated decoder IoU itself was still discarded, that is the part now resolved.)*
4. **QC post-hoc → in-run (M2).** `run_chain` now calls `pipeline.run_qc` right
   after `save_masks`, so QC + flagging happen as part of every run, headless.
   It still reads the just-written PNGs back rather than scoring *inside* the
   propagate loop, that fully-interleaved form is only needed for
   *halt-and-re-prompt*. The §3c generator restructure it depends on has now **landed**
   (`PropagationSession`), so the mechanism exists; wiring QC *into* the loop as an
   auto-intervention policy is the remaining **M4** step. So: "QC moved into the run,"
   not yet "into the loop."
5. **Anisotropy for Blender.** Voxels are 2/2/50 nm; at SCALE=8 that's ~16/16/50 nm
   (z ≈ 3× xy). Whatever meshes the volume must receive correct z spacing or the
   neuron will be squashed in z. *(Still open; M5.)*
6. **Per-chain skeleton for containment (M2, AVAL-surfaced).** The
   `skeleton_contained` probe must use *this chain's* skeleton nodes, NOT the whole
   neuron's. First AVAL run flagged 100% of frames because `compute_metrics` was
   handed the full neuron (`skeleton=annotate_df, cell_name="AVAL"`); AVAL is ~24
   chains, so its nodes cross each z at several xy and their centroid sits off any
   single process → never inside the mask, even at the anchor. Fix: `run_chain`
   filters `annotate_df` to `chain["nodes"]` before QC. Also, `skeleton_contained`
   is now **tri-state**, `True` / `False` / `NaN` (no chain node at that z; a
   non-monotonic neurite leaves the section), and only an explicit `False` flags,
   so the ~30% of frames in z-gaps abstain rather than false-flag. This is a
   concrete instance of the §4 "centralize coordinate transforms / tag the space"
   principle: the bug was a *reference-set* error (whole-neuron vs chain), invisible
   until eyeballed on AVAL.
7. **Manifest is append-mode; a scoring-threshold change mid-campaign silently mixes
   thresholds (M3.5, June 2026).** `_manifest.csv` rows are written/kept per chain as
   they run, not rewritten. If a gate threshold changes between runs (e.g.
   `gate_max_area_frac` 0.05 → 0.4), chains scored earlier keep their old-ceiling
   `anchor_passed`/`anchor_reasons` while new chains use the new ceiling, so
   `anchor_passed` silently reflects two configs at once. *Symptom:* `area`-FAIL rows
   whose `anchor_area_frac` is *below* the supposed ceiling (seen pre-clear: area-FAILs
   down to 0.055 alongside PASS rows up to 0.38). *Fix:* clear the manifest (or force
   re-score) after any scoring-threshold change; sanity-check uniformity by confirming
   min(`area_frac` among area-FAILs) > max(`area_frac` among PASS).

The three failure signals you listed already exist in `qc`:
`area_ratio` (size change), `temporal_iou` (overlap), `skeleton_contained` (node
containment), plus a composite flag/intervene rule. Auto-detection is mostly
**moving these inline** + tuning thresholds, not building from scratch.


---

<a id="old-7"></a>
## old §7, Design decisions: full log (landed + rejected, with rationale)

> Moved from design-notes.md §7 "Open decisions (not blocking)". The live doc now carries
> a crisp **§7 Design decisions & knob rationale** (decision + current default + one-line why,
> for the topics the README cites). This is the full original log, including the verbose
> "Update June 2026, landed…" annotations and the rejected alternatives. Read this for the
> reasoning and measurements behind any knob default.


*Tagged by milestone (see §6): **M3.5** = headless anchor-quality + calibration harness (pre-GUI); **M4** = napari GUI (collects labels); **M4.5** = predictor model + label-gated accuracy (consumes M4's labels, the learned `P(error)` detector, EM-finetuned SAM, and the tier-2 crop / `pred_iou` floor / `gate_max_area_frac` levers that need ground truth to tune); **M5** = aggregation → Blender. "auto · human→M4" means the automatic part lands in M3.5 and the human-interaction part stays in M4; "M4 logs · M4.5 trains" splits label collection (M4) from model training (M4.5). Untagged items are unscheduled.*

- **[M5]** **Blender import format:** raw PNG planes (simplest) vs. a single 3D label
  volume vs. pre-meshed `.obj/.ply` (marching cubes + decimation). Affects §5.5.
- **[M3.5]** **QC thresholds:** defaults (area_ratio ∉ [0.5, 2.0], temporal_iou < 0.3,
  pred_iou < 0.5) are now `qc_*` knobs on `PipelineConfig`, tune there, not in
  `qc.py`. First AVAL run: 40/104 flagged (38%), 5 intervene, with `skel miss: 36`
  (node present, mask doesn't cover it) vs `skel n/a: 31` (no node at that z). The
  36 containment misses are the next thing to eyeball, are they real drift, or
  is `qc_skeleton_dilation_px` (3) too tight for thin neurites? Tune before
  trusting the flag rate at scale.
  **Subset-run result (M3, June 2026, 24 neurons / 260 flagged chains / 3,789
  flagged frames):** `skeleton_contained=False` (`noskel`) drives ~90% of all
  flags, while `area_ratio` (median 0.99) and `temporal_iou` (median 0.67) look
  healthy on flagged frames, so the headline flag rate is currently a
  skeleton-containment artifact, not a degradation signal. Only 16% of flagged
  frames are intervene-level (≥2 signals), so plan around *intervene*, not raw
  flags. **Item-0 sweep (RAN, June 2026):** the automatic `qc_skeleton_dilation_px`
  sweep (0..10 px) over the saved masks showed the *intervene*-level queue is
  dilation-robust, `intervene_rate` moved <0.005 across the sweep, so the multi-signal
  queue does not hinge on the `noskel` threshold. What the sweep *cannot* say is which
  vanishing single-signal flags were real errors; that correctness call waits for
  M4-collected labels (manual gold-set shelved, see *Manual gold-set labeling* below).
  **Update (June 2026): `pred_iou` populated.** pred_iou now comes from SAM2's
  mask-decoder IoU head (§5 #3). Enabling it **changes the flag/queue distribution**, it's
  a real 4th signal at `qc_pred_iou_min` (default 0.5). Per the §5 #7 mixed-threshold
  discipline, **clear/re-score the manifest** on the first run with pred_iou on, or early
  chains (NaN, inert) silently mix with later chains (live). To record pred_iou *without*
  flagging on it (observe first, per the §6 measure-then-trust ruler), set
  `qc_pred_iou_min <= 0`. First task before trusting it at scale: read the pred_iou
  distribution on a clean run and confirm 0.5 is the right floor, it's a borrowed default,
  not yet calibrated on this data. **This calibration is now M4.5** (label-gated): the floor
  is set against ground-truth verdicts, and pred_iou is the strongest feature for the M4.5
  learned detector, so calibrating it belongs in the predictor-model milestone, not pre-GUI.
  Likewise the **`gate_max_area_frac` 0.4-vs-0.75 finalization** (below) moves to M4.5, it
  needs labels to choose; until then leave it at 0.4 and filter `area`-only FAILs by
  `anchor_contained`/`anchor_lcc`.
  **Crop A/B (M3.5, AVAL, June 2026, default crop on vs off, same 24 chains):** the
  anchor gate is now scored in `_crop`, which makes its thresholds **space-relative**, the concrete fallout of the §4 `scale` narrowing. (a) `area_frac` is measured against
  the crop, not the full frame, so it jumped ~35× (median 0.0004 → 0.014); the old
  `gate_max_area_frac = 0.05` ceiling mis-fires on clean thick AVAL anchors
  (`contained=True`, `lcc≈1.0`), producing spurious `area` FAILs. **Action: raise
  `gate_max_area_frac` for crop space** (the AVAL data put the floor at ≈0.4). (b)
  `gate_min_area_frac = 1e-5` is now **inert**, the small-cell false-positive worry is
  *resolved* by the higher resolution. (c) The containment radius and box margin are
  auto-rescaled by `scale/crop_scale` (×4) inside `run_chain`, so
  `qc_skeleton_dilation_px` stays the one knob and keeps anchor- and per-frame
  containment physically equal. Until the ceiling is set, filter FAILs by
  `anchor_reasons` (an `area`-only FAIL with `anchor_contained=True` and high
  `anchor_lcc` is a clean anchor, not a failure).
  **Clean 5-neuron run (M3.5, June 2026, 233 chains, AIAL/AIAR/AIYL/AIZL/AIZR, uniform
  `gate_max_area_frac=0.4` after the §5#7 manifest-clear):** the first trustworthy
  single-ceiling thin-neuron read (an earlier 0.05-ceiling partial batch was discarded
  as mixed-threshold, §5#7). (a) **Anchor 230/233 pass (98.7%); containment 233/233;
  median `anchor_lcc` ≥0.995 per neuron**, the seed half is solid across thin neurons.
  (b) **Area limb clean at 0.4:** one `area` FAIL, AIZR 7 (`area_frac` 0.66, `lcc` 0.99,
  contained, a legit-large clean anchor; would pass at 0.75). (c) **The 2 `frag` FAILs
  are borderline false alarms:** AIYL 34 (`lcc` 0.799) and AIAR 8 (`lcc` 0.761) sit at
  the 0.8 `min_largest_cc_frac` and both propagated clean (`flag_rate` 0), so `frag`
  alone should *flag*, not hard-gate. **Decision still open:** keep 0.4 (accept the rare
  legit-large `area` FAIL) vs raise to 0.75 (0 `area` FAILs, lean on containment +
  `frag`). (d) **The triage queue is still noskel-dominated and unmoved:** of 572
  flagged frames, `noskel` is present in 82% and is the *only* signal in 72%; only 80
  are `intervene` (29/233 chains `flagged`), the same 80-90% signature as M3 and the
  pre-clear 376-run, so crop + items 3/5 do not touch it, as forecast. The 14%
  intervene-vs-flag split is the item-4 case, now landed (the queue surfaces the 80
  `intervene` frames, not the 572; see item 4 below); whether the 82% single-signal `noskel`
  is real drift or a `dilation_px=3` artifact was the item-0 question, the sweep (item 0,
  RAN) showed the *intervene* core is dilation-robust, but the pure-`noskel` residue stays
  ambiguous pending M4 labels.
- **[M3.5, superseded by item 4]** **Chain-level verdict:** `qc_intervene_to_flag_chain` (default 1) marked a chain
  `flagged` when ≥1 frame hit `intervene` (≥2 signals); `triage_frames` originally
  listed every single-signal flag. Subset run (M3) confirmed the split matters: raw
  flagged chains (260) vastly outnumber chains with any intervene frame, because a
  single `noskel` flag marks a chain `flagged`. **Resolved by item 4 (below):** both
  the per-frame queue and the chain verdict now key on the same queue definition
  (`flag_count >= qc_triage_min_signals`, default 2 = intervene), so `triage_frames`
  no longer lists single-signal flags.
- **[M3.5, landed] Item 4, triage queue gated on `intervene` (June 2026).** Both the
  per-frame human queue and the chain verdict now surface only frames at/above a configured
  severity, `flag_count >= qc_triage_min_signals` (new knob on `PipelineConfig`, default
  **2 = intervene**; set to 1 for the legacy "queue every flag"). `run_qc` writes a `queue`
  column to `qc.csv` (so the cross-chain rollup filters on the artifact alone, §4) and adds
  `n_queue` / `queue_rate` to `qc_summary`; `n_flagged` is **retained**, single-signal flags
  stay on disk as diagnostics and as M4 label fodder, just not surfaced to a human.
  `triage_frames` and `_triage.csv` are now the queue (intervene) frames; chain `status`
  keys on `n_queue` so the two can never disagree. **Behaviour-preserving at defaults:**
  `min_signals=2` → `n_queue == n_intervene`, identical to the prior
  `n_int >= qc_intervene_to_flag_chain` rule. `batch.build_triage_queue` filters
  `queue → intervene → flag` (fallback chain), so a pre-patch run rebuilds straight to the
  intervene set with no re-segmentation.
  **Validation (clean 5-neuron run, 233 chains / 2082 frames):** rebuilding `_triage.csv`
  off the existing `qc.csv` (intervene fallback) drops it **572 → 80 queued frames**, exactly the item-0 sweep's d=3 `n_intervene=80`, with `_manifest.csv` statuses unchanged
  at **29 flagged** chains (the verdict was already intervene-gated). The 80 are
  *multi-signal-corroborated*, not `noskel` survivors: e.g. AIAL/0 z1548 = `area×3.9` +
  `tIoU 0.02` (a propagation runaway), and several queue frames have `skeleton_contained=True`,
  flagged purely on `area` + `tIoU`. This is the dilation-robust core item 0 identified, `intervene_rate` moved <0.005 across the 0..10px sweep.
  **Bound (→ §5 #3):** with `pred_iou` now populated (§5 #3) `flag_count` can reach 4, so the
  queue gains a 4th corroborator beyond the three *geometric* signals (`area_ratio`,
  `temporal_iou`, `skeleton_contained`). This directly attacks the failure mode this Bound
  originally called out, a stable-but-wrong frame (plausible area, good temporal overlap,
  node happens to fall inside the mask) tripped none of the three geometric signals, could
  not reach `intervene`, and stayed invisible; SAM2's decoder IoU is the signal most likely
  to surface it. (Pre-pred_iou, `flag_count` topped out at 3 and this case was uncatchable, that is the gap now closed in mechanism, still to be confirmed at scale once the pred_iou
  floor is calibrated, §7 *QC thresholds*.)
  **`review` left on `flag` by choice**, `grid_flagged` / `animate_flagged` still eyeball
  *all* flags (the "are the flags sane" diagnostic, the item-0 use case); the human *work*
  queue is `_triage.csv`. A one-line switch to `queue` is noted in the patch if that
  distinction ever wants collapsing.
- **[shelved]** **Manual gold-set labeling (`calibration.py`), dropped for now.**
  The plan was a read-only human labeling pass, the **anchor** as a chain-level
  gate, **all flagged frames** (precision), and a **uniform random sample of
  un-flagged frames** (to estimate the silent-error rate the queue can't show), producing ground truth to calibrate the `qc_*` rule. **Decision (June 2026):
  shelved**, too much manual effort up front. Ground-truth labels are deferred to
  M4's GUI logging (*GUI as label engine*, below); M3.5 measurement falls back to
  automatic proxies (anchor-gate pass rate + queue deltas at fixed thresholds).
  `calibration.py` stays parked in the repo, off the critical path. The two
  structural facts it encoded still hold and still constrain M4's logging:
  (a) anchor quality gates everything downstream, log the anchor verdict and keep
  bad-anchor chains out of training; (b) a detector can only be measured against
  truth, never its own flags, so the GUI *must* still log a random sample of
  un-flagged frames, or the eventual model stays blind to silent errors.
- **[M4 logs · M4.5 trains]** **GUI as label engine → learned QC detector (on-the-go model training).** The
  hand-tuned `qc_*` rule is the bootstrap; the successor is a learned `P(error)`
  model whose training data is the *exhaust* of M4 review. Every correction a human
  makes is a label, so M4 logs one flat per-frame row, the QC signal vector
  (features), human verdict + error_type, the chain's anchor verdict, the frame's
  role, and whether the *rule* flagged it, into a per-frame label store M4 owns
  (the schema `calibration.py` sketched, now that the manual tool is shelved), live.
  Payoff: replace four hand-thresholds with one sliding probability knob (a clean
  precision/recall dial). This **dissolves the GUI-vs-calibrate ordering**, the GUI
  *is* the calibration instrument, the gold set is its byproduct, and the model is
  judged by §4's only metric: does it shrink the triage queue. Considerations, each
  a way the naive version quietly fails:
  - **Selection bias is the killer.** Labels collected only on *flagged* frames are
    censored: the model can learn to distrust the rule (cut false positives) but can
    *never* learn to catch what the rule misses (silent errors), because it sees no
    examples of the stable-but-wrong regime. M4 must therefore log a random sample of
    *un-flagged* "good" frames too (the `sampled` role above), even though there's
    nothing to correct on them. Non-negotiable, without it the model only ever
    shrinks the queue, never widens coverage.
  - **Anchor contamination poisons labels.** A frame wrong because the seed was wrong
    is not a propagation-signal failure; training on it teaches the model to predict
    anchor failures from features that can't see anchors. Log the anchor verdict;
    exclude (or separately model) bad-anchor chains.
  - **Feedback loop / covariate shift.** Each time you tighten the operating point you
    stop seeing the frames you now suppress, so the training distribution drifts under
    you (a filter trained on its own filtered output). Guard with a held-out,
    randomly-sampled eval set the model's decisions never touch, the only honest
    "is it improving" signal.
  - **Features bound the model.** It can only catch errors expressible in the signals;
    the stable-wrong case needs richer features. `pred_iou`, likely one of the strongest
    features and the one that targets the stable-wrong case, is **now populated** (§5 #3,
    June 2026), so it is available to log from the first M4 frame; calibrating its floor on
    this data is the remaining step (§7 *QC thresholds*).
  - **Keep modeling boring.** Logistic regression or a small gradient-boosted tree
    over the handful of signals; data volume is not the constraint, label *coverage*
    is. Split train/test **by chain or neuron**, adjacent frames share temporal
    signals, so a frame-level split leaks and inflates accuracy.
  - **Active learning is the real payoff.** Once a model exists, let its *uncertainty*
    (P≈0.5) choose what the human labels next, instead of relabeling whatever the rule
    already flagged, far fewer labels for a better detector. Conservative rule =
    turn-1 bootstrap; model uncertainty = turn-2 labeler.
- **[M3.5 auto · human→M4]** **Anchor selection:** mid-frame is the current heuristic; should a failed anchor
  re-pick (e.g. next node toward the chain center) automatically before queueing
  a human?
  *(Update June 2026: the gate that would drive a re-pick now exists in scoring-only
  form, `score_anchor` records the verdict but does not yet act. The automatic re-pick
  remains the open piece; it moves out of the `# [M4]` markers into the gate when the
  gate goes from observational to acting.)*
- **[M5]** **Chain merge conflicts:** when two chains of one neuron disagree on a z-slice,
  is the aggregate a plain union, or does overlap/voting matter?
- **[M3.5]** **Anchor prompt quality + a pre-propagation gate (raised, skeptical, unmeasured).**
  First-pass image-mode anchors, a single skeleton point + a few neighbour-cell
  *centroid* negatives, on the SCALE=8 frame, are often poor on thin neurites (a
  process is only a few px wide at 8×, below what SAM reliably segments). Cheap
  levers that need *no* model change: (a) run image mode on a full-res / 2× **crop**
  around the node instead of the 8× full frame, then downscale the box/mask to SCALE
  for the video seed (decouples anchor precision from propagation VRAM); (b)
  synthesize a **box** prompt from the CATMAID `radius` column (already in the node
  table) rather than a bare point, a box encodes extent, a point doesn't; (c)
  `multimask_output=True` + auto-select by node-containment / plausible-area /
  single-CC. Separately, an **anchor-quality gate** *before* propagation (score the
  anchor → auto-escalate prompts / re-pick the node toward chain centre → queue a
  human only on repeated failure) would make a bad anchor cost one frame's compute
  instead of a wasted ~300-frame propagation. This is the concrete form of the
  *Anchor selection* item above. Treat as unproven: measure the auto-anchor success
  rate first (the M3 batch is the instrument) and, per §4, invest only if it shrinks
  the queue enough to pay for the code.
  *(Update June 2026: lever (a), the full-res crop, **landed as the default**, not as a
  gated escalation (see §6 M3.5 and §4). The gate itself (`score_anchor`) is wired and
  recording but still observational. Lever (c) `multimask_output` auto-select **landed,
  default-off** (`multimask_anchor`; `_select_anchor_mask` ranks the 3 candidates by
  node-containment → plausible-area → single-CC → decoder IoU). Verified near-free against
  the SAM2 source: the mask decoder computes all 3 candidate masks regardless of the flag
  and only slices the output (`sam2/modeling/sam/mask_decoder.py` `forward`), and the heavy
  image-encoder `set_image` runs once either way, so the "3× slower" worry does not hold;
  the only added cost is CPU-scoring 3 masks, and it touches only the one-frame anchor, not
  the ~300-frame video propagation. Lever (b) box-from-`radius` is **dead**, the CATMAID
  `radius` column is mostly placeholder values (decided June 2026).)*
- **[M4.5 model swap · napari-plugin eval→M4]** **EM-finetuned SAM / micro_sam, considered, deferred.** Domain-adapted SAM models
  segment EM neurites markedly better than vanilla SAM2 (the natural-image domain gap
  is real and well documented). Deferred anyway because: (1) it is *not* a localised
  swap, this pipeline's core is SAM2's **video** propagation, while micro_sam's
  finetuning targets the interactive **image** SAM, so it doesn't drop into
  `build_sam2_video_predictor`; at most it improves the anchor/image phase and forces
  a second model path. (2) It's still not 100%. (3) Per §4, accuracy isn't the
  bottleneck, supervised-per-chain already beats manual ~50×, so the refactor cost
  outweighs the gain right now. Decision: stay on vanilla SAM2; revisit only if
  measured failure rates make it pay. Orthogonal note: micro_sam's **napari plugin**
  (interactive EM prompting + annotation/correction, finetuned EM checkpoints) is a
  build-vs-adopt candidate for the **M4 GUI**, independent of any model swap, worth
  a look before writing a correction GUI from scratch.
- **[M3.5]** **Mask post-processing (new idea; cheap, deterministic, no model).** Saved masks
  are downscaled then nearest-neighbour upscaled, so they come out blocky / speckled
  / holey, whereas true neurite borders are smooth, so cleanup priors are safe to
  apply. A candidate `postprocess_mask` phase (§3a) between `propagate` and
  `save_masks`: largest-connected-component keep (generalise the one already in
  `box_from_mask`), morphological **open** (kill speckle) + **close** (bridge grid
  gaps), `binary_fill_holes`, and light boundary smoothing (morphological, or contour
  approximation / distance-field re-threshold). Open questions: apply at SCALE space
  (small kernels) or post-upscale (larger kernels); tune the kernel so thin neurites
  aren't erased (same failure mode as `qc_skeleton_dilation_px` set too tight);
  and decide ordering vs QC, clean *then* QC so QC scores the delivered mask, but
  watch that cleanup doesn't paper over a real propagation failure the QC signals are
  meant to catch.
  *(Update June 2026: **landed as a phase, default-off.** `pipeline.postprocess_mask`
  (model-free, `_sam`): morphological **open** (despeckle) → **close** (bridge NN-upscale
  grid gaps) → **largest-CC keep** (generalises `box_from_mask`'s pick) → `binary_fill_holes`.
  Decisions taken: (a) applied at SCALE/`_sam` space (small kernels, `postproc_open_px` /
  `postproc_close_px` default 1; a kernel bigger than the neurite half-width erodes thin
  processes, same failure mode as `qc_skeleton_dilation_px` too tight); (b) **ordered before
  QC**, folded into the save step so the saved PNGs are the cleaned masks and `run_qc`
  scores the delivered mask. Open caveat kept live: `postproc_keep_largest_cc` (default
  True) drops genuinely-split components in the *saved* mask and can mask a real propagation
  failure, so watch the flag distribution when enabling and consider False for chains where
  a process legitimately leaves/re-enters the plane.)*
- **[tier 1 DEFAULT · tier 2 LANDED default-off (June 2026) · tier 3 → later]** **Local
  high-res cropping (prior art: Bader Lab `sam2maskpropagator`).** Attacks the core accuracy
  problem (a neurite is ~3 px wide at scale 8). **Tier 1 (anchor-only crop) is now the default image phase**
  (June 2026): `run_chain` loads the full-res anchor frame, crops a `crop_size_tif`
  (default 1200 px) window around the node via `alignment.CropWindow`, runs image mode in
  `_crop` at `crop_scale` (default 2 → ~600 px input, near the old 8× cost but ~6× the
  linear resolution on the neurite), and maps the largest-CC box `_crop→_sam` for the
  video seed. The promised **single centralised transform** is `CropWindow`
  (`around_node` / `tif_to_crop` / `crop_to_sam` / `box_crop_to_sam` / `slice_tif`), which
  sidesteps the Bader x/y-swap trap (§4/§5), the row/col swap lives only in `slice_tif`,
  verified by a marker round-trip test. **Measured (AVAL + the clean 5-neuron run):**
  sharpens the seed (cleaner anchors, tighter `box_sam`) but leaves the downstream
  `noskel` queue ~unchanged, as expected, since an anchor-only crop does not touch
  propagation resolution. Compute note: the default path adds one full-res `imread`
  (~240 MB at ~9k²) per chain's anchor, freed after; a windowed/memmap tiff read is a
  later optimisation. **Tier 2, per-chain propagation crop, LANDED default-off (June 2026,
  supervisor-authorized accuracy-first; brought forward from M4.5 at the lab's request).** A
  new space `_pcrop`: `run_chain` crops ONE window sized to the chain's whole skeleton
  xy-extent (+ `chain_crop_pad_tif`, default 64) and runs the *entire* image phase **and**
  propagation inside it at `chain_crop_scale` (default 2), instead of the scale-8 full frame, the lever that actually moves downstream propagation resolution (tier 1 only sharpened the
  seed). Knobs on `PipelineConfig`: `chain_crop` (master switch, **default False** → the _sam
  full-frame path and the M1 baseline are unchanged), `chain_crop_pad_tif`, `chain_crop_scale`
  (a *target*, bumped coarser per chain so the input's longest edge stays ≤ `chain_crop_max_px`,
  default 1536, bounding VRAM for a chain that wanders far) and `chain_crop_min_tif` (default
  1024, a **floor** on the window extent: a low-motion chain whose xy-bbox is tiny otherwise
  over-zooms and SAM2 loses inter-frame context; see the A/B below). Implementation reuses the
  single `CropWindow` home (new `around_box` builder + `sam_to_crop`; the only `[y,x]` swap
  stays in `slice_tif`). **Masks are stored in `_pcrop`** (the resolution win is kept on disk, for
  Blender), and the `CropWindow` is persisted to `state.json` (`ChainState.crop_window`) so
  QC, `review`, and the GUI rebuild the crop space: `qc.compute_metrics` maps skeleton nodes
  `_tif→_pcrop` via the window (the `scale==save_downscale` guard is skipped in crop mode, the
  containment radius rescaled by `scale/crop_scale`), and the napari GUI reconstructs the
  window for skeleton overlay + `--hires-em` (everything it shows, frames, masks, clicks, already shares the `_pcrop` grid, so a click is a `_pcrop` coord and re-predict/resume need
  no transform; this also incidentally gives tier-2 chains the **crop-space re-predict** the
  M4 GUI had deferred). Frame prep (`prepare_chain_crop_frames`) crops each frame the SAME
  crop-then-downscale way as the anchor, so seed and propagated frames share exact `_pcrop`
  pixels; it loses the cross-chain decode cache (each window is unique → one full-res imread
  per frame; windowed/memmap read is the documented follow-up). Window math + crop-aware QC
  are unit-tested (`tests/test_alignment.py`, +4 cases; 17/17). **A/B MEASURED (June 2026,
  real SAM2, RTX 3050 6GB, `ab_tier2.py` harness, 3 AIYL chains, large model, tier-2 on vs
  off):** tier-2 moved **2/3 chains `flagged`→`done`**, c12 (3 queued→0), c29 (3 queued→0,
  5 noskel→0), at the crop's higher resolution (masks ~512px in `_pcrop` vs a ~3px-wide
  neurite speck at scale-8; ~17-30× the foreground pixels describing the same process), and
  the 3rd chain (c02, already clean) stayed clean, **no regression once the min-extent guard
  was in**. *The guard came directly from this A/B:* an **un-guarded first pass catastrophically
  failed c02**, a low-motion neurite (tiny xy-bbox) produced a 156×244 over-zoomed window where
  the anchor scored 0.52 and propagation collapsed to **empty masks on 29/39 frames** (→ 30
  queued). Adding `chain_crop_min_tif=1024` (floor the window, pad out for context) recovered it
  to anchor 0.90 / 0 empty / `done`. **So: tier-2 is a strong per-chain lever (eliminates the
  `noskel` queue on chains that have xy-motion) but NOT safe to enable blindly, over-zoom on
  low-motion chains is a real failure mode; the min-extent floor mitigates it, and the
  `image_score`/anchor gate should guard a per-chain fall-back to `_sam` (next step).** Cost:
  tier-2 ran ~2× the wall-time of baseline here, dominated by the per-frame full-res `imread`
  in frame-prep (propagation itself is *cheaper* than the full frame) → the windowed/memmap
  read is the priority optimisation. Verified non-degenerate + visually (overlay grids in
  `ab_figs/`). **Open next:** (a) image_score/anchor-gated auto fall-back to `_sam` when a crop
  anchor is poor; (b) tune `chain_crop_min_tif` (1024 slightly relaxed c12's tight win, 0 queued either way); (c) the windowed/memmap frame read; (d) wider A/B across neurons +
  M4-label confirmation that the remaining single-signal `noskel` is benign.
  *(Update June 2026, items (a) and (c) landed; ab_fallback.py.)* **(a) anchor-gated fall-back to
  `_sam`** is in (`chain_crop_fallback`, default on): when a chain's `_pcrop` anchor is poor it
  re-runs the whole chain in the plain `_sam` path instead of propagating a collapsed crop.
  **Key correction from the A/B:** the geometry gate alone does NOT catch the over-zoom, the
  over-zoomed anchor *passes* it (clean blob, contains the node); the collapse is a propagation
  effect, invisible at the anchor frame. The discriminating signal is SAM2's anchor `image_score`
  (over-zoom **0.516** vs healthy **0.848 / 0.879**), so the fall-back fires on
  `chain_crop_min_image_score` (default **0.7**, first-pass, tune in the wider A/B). Verified: the
  forced-over-zoom chain falls back and recovers the clean baseline (status `done`, 0 queued, _sam
  dims) at no extra wall-time; a good-crop chain (0.879) stays tier-2 and does not regress
  (`fell_back_to_sam` recorded on ChainState for the P(error) features). **(c) windowed/memmap
  frame read** is in (`_read_tif_window`): a `tifffile.memmap` row-window slice (the EM tifs are
  uncompressed single-strip 8-bit grayscale → memmappable), with a full-`imread` fallback for any
  tif that isn't. Measured **bit-identical** to `cv2.imread(tif)[sl]` and **~48× faster** per frame
  (566→12 ms), so the ~2× tier-2 wall-time penalty above is essentially gone. Still open: (b) tune
  `chain_crop_min_tif`; (d) wider A/B across neurons + the `image_score` floor's true value.
  **Tier 3 → later:**
  (3) **per-frame tracked** crop following skeleton xy(z), max resolution, but a shifting origin
  per frame (per-frame remap; may help by centring the object, may confuse tracking, speculative); still unbuilt.
- **[M3.5 auto seed · human anchor→M4]** **Video seed: box vs mask (confidence-gated), incl. human-painted anchors.** The box
  doesn't *avoid* needing an accurate anchor mask, it *delegates* making one to SAM2:
  a box seed has SAM2's decoder produce the anchor mask and stores *that* in the memory
  bank, and that box→mask step is exactly the single-image guess we've shown is
  unreliable on thin EM neurites at 8×. A mask seed (`add_new_mask`) bakes a curated
  boundary into memory instead, strictly more informative *when it's right*, but memory
  propagates faithfully, so a slightly-wrong mask propagates its error whereas a box
  lets SAM2 re-derive something plausible. Hence a **confidence gate**: seed with the
  mask when the anchor is trustworthy (QC-pass / high `image_score` / human-touched),
  else fall back to the box. Per §4 (accuracy + HITL over automation-%), the mask is the
  *target* seed; the box is the transitional default while anchors are still auto-and-
  rough. Nearly free to try, `image_predict` already computes the mask and the current
  pipeline discards it for its bbox, so just add a mask-seed path in `propagate` + the
  gate. A **human-painted anchor** is the maximally-verified case: for tiny / single-node
  / E-or-U chains where prompting fights you, the human paints the anchor and SAM2
  propagates the rest, automating the ~300-frame step while conceding the one hard
  frame. Per-chain routing: trivially small chains, or chains whose anchor QC fails after
  auto-retries, go to human-anchor rather than burning compute. Folds into the same M4
  mask-edit surface, and directly serves the supervisor's accuracy + HITL mandate (§4).
  *(Decision June 2026: **co-build this with the M4 GUI, do not build it pre-GUI.** Two
  reasons: (1) its highest-value mode, the human-painted anchor → mask seed, *is* an M4
  feature (the mask-edit surface), and (2) certifying "when does the mask seed beat the box
  seed" needs ground-truth labels, which only the GUI produces (§6 ruler; the M3.5 proxy
  ruler can't see silent errors). The mechanism is already in place, `PropagationSession.add_mask`
  exists and is AVAL-validated, so the auto/confidence-gated path is a thin add alongside
  the GUI's human-anchor path, not a separate build. This is the counterpart to the multimask
  decision, which we landed pre-GUI precisely because it is near-free and self-contained.)*
  *(Update June 2026, **seed ablation landed + measured** (`ab_seed.py`; flexible
  `seed_box`/`seed_points`/`seed_negatives`/`seed_mask` + `box_margin_frac`). **API fact:** SAM2
  makes MASK and POINTS/BOX mutually exclusive per frame (`add_new_mask` pops `point_inputs` and
  vice-versa), so "mask + points on the anchor" is NOT a real config, the valid space is
  mask-only OR any subset of {box, pos, neg}. **Result (3 AIYL chains, 88 frames, anchor held at
  scale-8 `_sam` so seed type is isolated), ranked by queue:** `box_pos` (the current default) = 6
  queued, tied with the box+neg / boxfrac variants; `box_only`/`mask_only` = 8; `pos_only` = 9.
  **Takeaways:** (1) **box+positive is the best AUTO seed, keeping the box was correct**; point-only
  *regresses*. (2) **`mask_only` does NOT beat the box at scale-8** (8 vs 6), confirms the mask seed
  only wins on a *high-quality* anchor (curated/human-painted or tier-2 `_pcrop`), so scrapping the
  box in the *GUI* (human paints a good mask) was right AND keeping it for scale-8 AUTO was right;
  the two decisions are consistent, not contradictory. (3) **Negatives are chain-dependent**, not a
  blanket win: c12 queue 4→1 with negatives, but c29 2→5, helps concave/cluttered, hurts clean,
  net wash. Keep `seed_negatives` a targeted lever, default-off. (4) **`box_margin_frac` (underfill
  fix), VALIDATED** (`ab_underfill.py`, scan 23 chains -> A/B the top-3 high-noskel suspects).
  **RIML c25 was a genuine underfill**: fixed-10px box -> noskel 9/21, queue 4, *flagged*;
  `box_margin_frac=0.5` -> **noskel 0, queue 0, *done*** (the size-relative pad enclosed the whole
  cell the fixed box clipped). Your bounding-box instinct was right, under-filled anchors are a
  real failure mode and the frac margin fixes them. BUT it's TARGETED, not universal: of 3
  high-noskel suspects only RIML c25 was true underfill; AIYL c12 (noskel identical across all
  seeds) was tracking drift and AVBR c12 (img_score 0.27) a poor anchor, frac was inert on both,
  and `mask_only` was WORSE on all three (re-confirming the mask seed needs a good anchor). So keep
  `box_margin_frac` default-OFF and make it a **targeted retry lever**: a chain that flags with high
  noskel + a contained anchor is the signal to re-run it with the frac margin (the same
  retry-on-failure pattern as the item-b tier-2 fallback). Caveat: small sample, weak deltas
  (6 vs 8-9), directional. Default seed unchanged (`box_pos` won); the knobs are additive.)*
  *(Update June 2026, **wider tier-2 A/B** (`ab_tier2_wide.py`, 15 chains × AIYL/RMDR/AVBR, tier-2
  with the item-b fallback on): improved 3, **regressed 0**, unchanged 12, fallback fired 6/15, net
  queue −10. Tier-2-with-fallback only helps or stays neutral (via fallback) across 3 neurons → safe
  to enable on flagged chains; AVBR c12 (worst, queue 9) fell back to `_sam` and needs the GUI.)*
- **[M3.5 auto · manual→M4]** **Negative points in video seeding.** `add_new_points_or_box` takes labelled points
  *and* a box on the prompt frame, so adding negatives to the video seed is trivial.
  Most useful for concave shapes (E/U neurons) where a box bounds a concavity that
  belongs to a neighbour; auto-negatives can come from neighbour skeleton nodes (same
  source as image mode). Same mechanism the M4 GUI uses to correct a degrading frame.
  Open: whether neighbour-node negatives actually land in the concavities, measure.
  *(Update June 2026: **landed, default-off.** `propagate(..., seed_negatives=False)` /
  config `seed_negatives`. On = forward the same-z neighbour negatives `build_prompts`
  already computes in `_sam` to the video seed (the seed-time analogue of the image-mode
  negatives); off = positives-only = the M1 seed. The "do negatives land in the concavity"
  question is now an A/B switch, still unmeasured. Risk to watch: the k-nearest negatives
  can include the same neuron's other chains / nearby branches, so on concave E/U chains
  confirm they suppress the *neighbour*, not legitimate foreground.)*
  *(Measured June 2026, see §8.3: negatives are chain-dependent (c12 4→1, c29 2→5), net wash, so
  they stay default-off as a targeted lever rather than a blanket seed.)*
- **[M3, landed]** **Runtime telemetry, landed.** `run_chain`'s `_step` now wraps each phase in
  `perf_counter`, accumulating per-phase seconds into `state.phase_seconds`; the batch
  driver brackets each chain with `diagnostics.reset_peak_vram()` / `peak_vram_gb()`
  (peak `torch.cuda.max_memory_allocated`) and appends one row per chain to
  `output/_timing.csv`, `neuron, chain_idx, n_frames, peak_vram_gb, t_<phase>…, t_total`.
  Fixed phase-label schema so appended rows never misalign; the write is wrapped so a
  telemetry hiccup can't kill a chain; review functions stay untimed (human-paced).
  Placement note vs. the original plan: the timer lives in `pipeline.run_chain` on
  stdlib `perf_counter` (keeps `pipeline.py` torch-free, same reasoning as the
  `on_video_phase` callback), `diagnostics` owns only the VRAM probes, and the batch
  driver owns the CSV. This is the prerequisite the speed items below were waiting on:
  one overnight run now yields time-vs-`n_frames` per phase (expect `propagate` to
  dominate, ~linear in frames now that prep is cached) and per-chain VRAM high-water
  marks (the headroom number for the GPU / multi-GPU questions). *(No longer open, just read the numbers off the next batch before any speed/hardware work.)*
- **[infra · parallel-review→M4]** **Performance scaling (GPU / multi-GPU chain sharding).** Propagation is GPU-compute
  + VRAM bound and sequential *within* a chain (the memory mechanism), so a faster GPU
  with more VRAM helps directly (more VRAM → less CPU offload). The big lever: chains
  are independent, so multi-GPU **chain-sharding** is the clean scale-out, each worker
  atomically claims a `pending` manifest row, runs it, marks it done; the resume design
  is already most of a work queue. Single-GPU multiprocessing mostly contends for one
  card, skip. Caveat (per §4 + supervisor): with a human reviewing flagged frames the
  human is the throughput limiter, so GPU speed shortens the *unattended* pass, not
  wall-clock to a finished dataset. Measure (telemetry) before buying hardware or
  building a multi-GPU harness.
  *Parallel review + background compute:* run the batch and the review GUI
  concurrently, background works `pending` rows (producer: segment + flag), the human
  works `flagged` rows (consumer), manifest = the shared queue. Wall-clock becomes
  max(GPU, human) instead of the sum, and the human is never blocked: by the time a
  reviewer clears the current flagged batch, more chains are done and freshly flagged.
  Tightens the test/refine loop too. Needs three things, all filesystem-only (no
  server/db): (a) **concurrency-safe manifest**, partition ownership (background owns
  *execution* status pending→running→done/flagged/failed; GUI owns a separate *review*
  status column) + a file lock (`filelock`/`portalocker`) around writes; (b) the GUI
  **polls/watches** the queue so chains flagged mid-session appear; (c) **GPU
  arbitration** for interactive re-runs (a human correction → `add_new_mask` →
  re-propagate competes for the card), interleave on one GPU (corrections are
  intermittent), or, under multi-GPU sharding, dedicate one GPU to interactive and the
  rest to batch. So parallel-GUI and multi-GPU are the same architecture from two angles.
- **[M4.5-ish] Marking/intervention GUI split (review-testing feedback, June 2026).** The
  single dense panel is confusing and lets the reviewer scroll to any frame and act on it,
  which muddies what the system thinks is being reviewed. Proposed two-mode flow: a
  **marking** mode that loads a chain and lets the human sweep frames ok/bad (label-only,
  no edits), and a separate **intervention** mode entered on a bad frame that shows *only*
  the flagged/selected frame(s) and exposes the correction tools (points / paint / resume).
  Cleaner than the current all-in-one dock and removes accidental-edit-while-scrubbing.
  Not built, the current single-panel GUI works; do this before or with M4.5 at the lab's
  discretion. Pairs naturally with the §6-row-4 deferred work.
- **[before M4.5] Strict-by-default flagging (review-testing feedback, June 2026).** Operating
  posture, not a mechanism change: set the `qc_*` thresholds to flag **aggressively** (high
  recall, catch every plausible error, tolerate false alarms) for the first labeled
  campaign, then loosen once the M4.5 learned `P(error)` detector has labels to set the
  operating point against ground truth (the *GUI as label engine* item). Concretely: tighten
  `qc_area_ratio_bounds` (e.g. `(0.7, 1.5)`), raise `qc_temporal_iou_min` and `qc_pred_iou_min`,
  and set `qc_triage_min_signals = 1` (queue every flag, not just intervene). NB the §5#7
  mixed-threshold discipline: **clear/re-score the manifest** after changing thresholds, or
  early and late chains silently mix two configs. Until the learned detector exists, the
  rule's job is recall, not precision, the human is the precision filter, and every
  decision is a label.


---

<a id="old-8"></a>
## old §8, M4.5 A/B results & decisions log

> Moved verbatim from design-notes.md §8. This is the canonical record of the M4.5
> measurement round, what was tried, the numbers, what was decided, and what was rejected
> and why. Kept intact because it is the primary source material for write-ups/reports.


The full record of the M4.5 measurement round, **what we tried, what the numbers were, what we
decided, and (just as important for future reports) what we REJECTED and why.** Ruler per §6:
relative review-queue deltas at fixed thresholds, not absolute correctness. Throwaway harnesses
(`ab_fallback.py`, `ab_seed.py`, `ab_tier2_wide.py`, `ab_underfill.py`) are committed for repro;
their `*.log` outputs are scratch. Commits: `0155c2b` (items b+c), `8f00330` (seed ablation),
`2cf448e` (underfill). All measured on the RTX 3050 / `large` model.

### 8.1 Item (b), anchor-gated fall-back tier-2 → `_sam`  ·  LANDED, default ON
**Goal (safety):** make `chain_crop` safe to enable broadly, a chain with a poor per-chain crop
should not propagate a collapsed `_pcrop` mask. **A/B (`ab_fallback.py`):** forced the c02 over-zoom
by dropping `chain_crop_min_tif` to 1.

| run | result |
|---|---|
| over-zoom, gate-only fallback (floor 0) | did NOT fall back, collapsed (95% flagged, 156×244) |
| over-zoom, `image_score` floor 0.7 | **fell back → recovered baseline** (done, 0 queued) at no extra wall-time |
| good crop (score 0.879), floor 0.7 | did NOT fall back, kept tier-2, no regression |

**Decision:** `chain_crop_fallback=True`, `chain_crop_min_image_score=0.7` (first-pass).
**Key finding / why the obvious approach was REJECTED:** the over-zoom collapse is a *propagation*
effect, the over-zoomed anchor *passes* the geometry gate (clean blob, contains node), so a
**gate-only criterion was rejected as insufficient** (it never fires). SAM2's anchor `image_score`
(over-zoom 0.516 vs healthy 0.85+) is the discriminating pre-propagation signal. **Also rejected:**
(a) *flag the chain* on a bad crop, wastes the chain vs. just using the working `_sam` path;
(b) *post-propagation* fallback (re-run in `_sam` only after QC sees the collapse), costs a full
~300-frame propagate before deciding, vs. the score floor which decides on one anchor frame.

### 8.2 Item (c), windowed memmap frame read  ·  LANDED, default
**Goal (perf):** kill the ~2× tier-2 wall-time from the full-frame `imread` in
`prepare_chain_crop_frames`. **A/B:** `_read_tif_window` vs `cv2.imread(tif)[sl]` over sample frames.
**Result:** **bit-identical** (the correctness bar for a pure-perf change) and **~48× faster**
(566→12 ms/frame). **Decision:** landed as the default read (EM tifs are uncompressed single-strip
8-bit grayscale → memmappable; full-`imread` fallback retained for any tif that isn't). **Nothing
rejected**, full-imread was the prior baseline; this strictly dominates on this data.

### 8.3 Seed ablation, mask vs box vs points vs negatives  ·  defaults UNCHANGED (`box_pos`)
**Goal (quality / "find the sweet spot"):** the lab asked whether scrapping the bounding box was
right, and noted *more prompts ≠ better*. **API fact baked into the harness:** SAM2 makes MASK and
POINTS/BOX **mutually exclusive per frame** (`add_new_mask` pops `point_inputs` and vice-versa), so
"mask + points on the anchor" is **not a real config**, the valid space is mask-only OR any subset
of {box, pos, neg}. **A/B (`ab_seed.py`, 3 AIYL chains, 88 frames, anchor held at scale-8 `_sam` to
isolate seed type), ranked by total queue (lower=better):**

| seed | queue | note |
|---|---|---|
| **box_pos** (default), box_pos_neg, boxfrac_pos, boxfrac_pos_neg | **6** | box+positive family wins |
| box_only, boxfrac_only, mask_only, pos_neg | 8 | |
| pos_only | 9 | worst |

**Decisions:** default seed stays **`box_pos`** (it won); `seed_negatives` stays **default-off**;
`seed_mask` stays **default-off**. The new knobs are additive (defaults reproduce M1 exactly).
**Rejected alternatives & why:**
- **mask-seed as the AUTO default, REJECTED.** `mask_only` (8) lost to `box_pos` (6) at scale-8.
  The mask seed bakes a curated boundary into memory, which only helps when that boundary is
  *right*; a scale-8 anchor is a coarse ~3px speck, no more informative than its box. This is
  **why both past decisions were correct and consistent**: keep the box for AUTO (scale-8), but the
  GUI *did* drop the box because there the human paints a high-quality mask, the one regime where
  the mask seed wins. The mask seed's home is the GUI / tier-2 `_pcrop`, not scale-8 AUTO.
- **negatives as a default, REJECTED.** Net wash in aggregate (6 vs 6) hiding *opposite* per-chain
  effects: c12 queue 4→1 *with* negatives, c29 2→5. Helps concave/cluttered, hurts clean → keep it
  a targeted lever, not a blanket default. (Confirms the lab's "more prompts ≠ better".)
- **point-only / box-removal, REJECTED.** `pos_only` (9) is the worst; dropping the box regresses.
- *Caveat:* small sample, weak deltas (6 vs 8-9), directional, not definitive.

### 8.4 `box_margin_frac` (%-of-bbox box pad, the underfill fix)  ·  VALIDATED, default OFF (targeted lever)
**Goal:** the lab's stated reason for revisiting the box, *"if the mask under-fills, a fixed box
doesn't enclose the whole cell."* **A/B (`ab_underfill.py`):** scan 23 chains/5 neurons for the
underfill proxy (anchor *contained* but high per-frame noskel), then A/B fixed vs frac vs mask on the
top-3 suspects.

| suspect | fixed box | **frac box (0.5)** | mask | |
|---|---|---|---|---|
| **RIML c25** | noskel 9, queue 4, *flagged* | **noskel 0, queue 0, *done*** | noskel 13 | **frac fixes a real underfill** |
| AIYL c12 | 12 | 12 (inert) | 12 | tracking drift, not underfill |
| AVBR c12 | 15 | 15 (inert) | 18 | poor anchor (score 0.27), not underfill |

**Decision:** `box_margin_frac` is **validated but stays default-OFF** as a **targeted retry lever**, a chain that flags with high noskel + a contained anchor is the trigger to re-run with the frac
margin (same retry-on-failure pattern as item b). **Rejected:** (a) **frac as a universal default, REJECTED**: inert on the 2 non-underfill suspects and `boxfrac`≈`box` on all non-underfill ablation
chains, so it offers no broad gain and could over-pad; (b) **mask seed for underfill, REJECTED**:
worse than fixed on all 3 suspects. **Confirms the lab's instinct:** under-filled anchors are a real
failure mode and the size-relative margin fixes them, so the box stays viable (with this lever)
rather than being scrapped.

### 8.5 Item (d), wider tier-2 A/B  ·  DECIDED → LANDED as an auto second-pass (§8.8)
**Goal:** decide whether flagged chains run tier-2 by default. **A/B (`ab_tier2_wide.py`, 15 chains
× AIYL/RMDR/AVBR, tier-2 with the item-b fallback on):** **improved 3, regressed 0, unchanged 12,
fallback fired 6/15, net queue −10.** Tier-2-with-fallback only helped or stayed neutral (via
fallback), zero regressions across 3 neurons. **Decision input:** safe to enable on flagged chains.
AVBR c12 (worst, queue 9) fell back to `_sam` and needs the GUI, tier-2 can't rescue a chain whose
crop anchor is also poor. **DECISION (June 2026, lab-approved): YES, flip tier-2 default-on for
flagged chains, as an automatic batch second-pass.** Implemented in `batch.py`; see §8.8 for the
mechanism, trigger semantics, and what verification covers it (the A/B above is the segmentation
evidence, the landed code only automates the trigger).

### 8.6 GUI manual-paint resolution  ·  finding, no code change
**Finding:** the paintable mask lives at the propagation resolution (scale-8 for `_sam` chains), and
`hires_em` sharpens only the EM *background*, not the mask, so hand-painted strokes on a scale-8
chain are 8×8-blocky when upscaled. This is inherent (`add_new_mask` must match the frame resolution
SAM2 propagates). **Decision:** keep manual paint; **crisp paint requires a higher-res propagation
space = tier-2**, its `_pcrop` view *is* the high-res paint surface, so "crisp manual paint" and
open item (e) (re-propagate a corrected `_sam` chain as tier-2) are the same lever.

**How this interacts with the §8.8 tier-2 auto second-pass (June 2026).** The GUI already does
*everything* in the space the chain was saved in: `open_chain` reads `crop_window` from
`state.json`, points `frames_dir` at the `_pcrop` view, and the mask/EM/prompt layers + clicks +
`add_mask` + the resume's `save_masks`/`run_qc` are all `_pcrop` (`gui.py` `_cw` path). So once
§8.8 lands a flagged chain in `_pcrop` *before the human opens it*, manual paint is crisp
**automatically**, the §8.6 "crisp paint = tier-2" requirement is satisfied by the batch, with no
GUI change. Concretely the two lab decisions are **the same "flagged → tier-2" policy in two
drivers**: §8.8 (batch) covers the bulk before review; (e) (GUI, deferred, §8.7) is the on-demand
version for a chain *still* in `_sam` when opened. After §8.8, the only `_sam` chains reaching the
GUI are ones that **fell back** (poor crop anchor → (e) would likely just fall back again) or were
flagged *in-GUI* after a correction, which is exactly why (e) was deferred, not why it's worthless
(see §8.7).

### 8.7 Deferred (recorded with the reason)
- **frac-margin auto-retry lever**, implement the §8.4 trigger (high-noskel+contained flag → retry
  with frac). Cheap; mirrors item b. Not built this round (validated the mechanism first).
- **~~Flip tier-2 default-on for flagged chains~~, LANDED (June 2026), see §8.8.** Built as an
  automatic batch second-pass, not a config default.
- **(e) GUI re-propagate-corrected-`_sam`-as-tier-2 (the crisp-manual-paint lever), DEFERRED, but
  LIVE: do NOT drop this in future iterations.** (Lab decision June 2026: defer, but keep flagged as
  relevant.) Rationale for deferring *now*: §8.8 lands flagged chains in `_pcrop` before review, so
  the common path already paints crisp (§8.6 interaction note); the chains (e) would still serve are
  the narrow set that reach the GUI in `_sam`, fell-back chains (poor crop anchor, so (e) likely
  falls back again) or chains flagged in-GUI after a correction. **Why it stays on the radar:** (i)
  those in-GUI-flagged and any non-fallback `_sam` chains have *no* crisp-paint path without it; (ii)
  if the §8.8 fallback rate turns out high (many flagged chains stuck in `_sam`), (e) becomes the
  primary crisp-paint route, not a niche one. **Build sketch when revived:** GUI button → build the
  chain crop (`pipeline.chain_crop_window`) → prep `_pcrop` frames (`pipeline.prepare_chain_crop_frames`)
  → map the current `_sam` painted mask/prompts into `_pcrop` → rebuild the `PropagationSession` over
  the `_pcrop` frames → re-propagate → migrate `masks/`+`state.json` (set `crop_window`) so the chain
  is now a `_pcrop` chain on disk (after which the existing GUI `_cw` path takes over). Consider a
  no-fallback variant (honor the human's explicit request even on a low-score anchor). Tracked also
  in §2 status and §6 row 4.5.
- **Stricter flag params**, see the strict-by-default bullet in §7; shelved until the learned
  `P(error)` detector has labels to set the operating point against (else we hand-tune twice).
- **Split marking/intervention GUI**, see the §7 bullet; not built, current single panel works.

### 8.8 Decision: tier-2 auto second-pass for flagged chains  ·  LANDED (June 2026)
**Decision (lab-approved):** flip tier-2 default-on for flagged chains (§8.5), implemented as an
**automatic batch second-pass** rather than a global `chain_crop` default, the first pass stays the
cheap `_sam` full-frame path for the thousands of clean chains, and only chains QC *flags* pay for
tier-2.

**Mechanism (`batch.py`).** `_run_one_chain` runs the chain once in `_sam`; if the result is
`flagged` and the new `tier2_on_flagged` knob is on (default **True**; module constant
`TIER2_ON_FLAGGED`, `run_batch` kwarg), it re-runs the chain **once** with
`replace(cfg, chain_crop=True)` and keeps that result unconditionally. The trigger is the pure,
unit-tested `_should_tier2_rerun(status, cfg_chain_crop, tier2_on_flagged)` =
`tier2_on_flagged and not cfg_chain_crop and status == FLAGGED`.

**Why it's regression-free.** Tier-2's own item-b fallback (`chain_crop_fallback`, §8.1) reverts a
chain with a poor crop anchor (`image_score < 0.7`) to the `_sam` path *within* the second pass, so
a kept-tier-2 chain only happens when the crop anchor is trustworthy, and a bad-crop chain lands back
exactly where the first pass left it. This is the §8.5 result (improved 3 / regressed 0 / net −10)
turned on by default. The second pass's `save_masks` overwrites the first pass's PNGs in place (same
chain z-range → same `mask_<z>.png` filenames), so no orphans; `diagnostics.cleanup_vram()` runs
between passes.

**Semantics / gotchas.**
- **Once per chain, same invocation only.** The second pass fires immediately after the first pass
  flags. A chain already `flagged` *on disk from a prior run* is skipped by `_should_run`
  (`flagged ∈ COMPLETE_STATUSES`), so it is never double-upgraded. **To upgrade a pre-existing
  flagged backlog** (e.g. the M3.5 5-neuron run) to tier-2, re-run those chains with `force=True`
  (or `clean`).
- **A still-flagged tier-2 chain is kept, not reverted to `_sam`.** Per §8.5 it's no worse than the
  `_sam` version and the human gets crisp paint (§8.6), the goal is `_pcrop`, not necessarily a
  cleared flag.
- **Timing.** `_timing.csv` `phase_seconds` reflects the *second* (tier-2) pass; peak VRAM spans
  both (reset before the first pass, read after).

**Verification.** (1) §8.5 is the **segmentation** A/B, the landed code invokes that identical
validated path, so no new GPU A/B is warranted for the *result*; it only automates the trigger.
(2) The trigger logic is unit-tested torch-free: `tests/test_tier2_rerun.py` (7 cases, the truth
table incl. the `not cfg_chain_crop` / flagged-only subtleties + the `replace` override). (3) **Still
to run live:** a single-chain integration smoke on a known-flagged chain confirming the second pass
fires end-to-end and writes a `_pcrop` chain (or falls back cleanly), not yet run, cheap to do on
the next batch launch.


### 8.9 Node-anchored multimask selection  ·  LANDED (code), measurement pending (June 2026)
Adapts the 2025 lightweight-SAM2 paper (Bhat et al.): the correct mask is usually among SAM2's three
candidates even when it is not the top-confidence one, so select it with an external anchor instead of
trusting the score. Their anchor is a Hoechst nucleus centre; ours is the skeleton node.

**What landed.** The selector `pipeline._select_anchor_mask` (already present, ranking `(contains the
positive node, plausible area, single-CC, SAM IoU)`, gated by `multimask_anchor`) was off in every
Stage-0 run. Turned it on for the `eval` preset. Added `multimask_exclude_neg` (default off): among
candidates containing the positive node, prefer one containing none of the negative neighbour nodes,
the anti-bleed pick (a mask that swallows a neighbour's node is bleeding). The two flags are separate
so A (selection) and B (negative-exclusion) measure independently; with `multimask_exclude_neg` off the
ranking is byte-identical to before. New pure helper `_negative_points`; the flag threads through
`image_predict` and `anchor_crop_predict` (negatives are already remapped to `_crop` space). See
[ADR 0012](adr/0012-node-anchored-multimask-selection.md).

**Verification.** Torch-free unit tests in `tests/test_anchor_select.py` (4 new cases: exclude-neg beats
higher IoU; exclude-neg off reproduces the old ranking; graceful when all candidates hold a negative;
positive-containment still outranks negative-exclusion). **Live A/B pending (user-run GPU):** score the
multichain set (AVAL/GLRDR/IL1L/URYVL) single-mask (the existing `out_gt_multichain` baseline), then A
(`multimask_anchor`), then B (`+ multimask_exclude_neg`), comparing precision / VOI_merge / micro-IoU.
The lever is justified only if the bleed metrics improve.

### 8.10 Pipeline core split into a package  ·  LANDED (June 2026)
The library core had grown to a ~2,200-line `pipeline.py` holding eight concerns at once. Split it into
a `pipeline/` package by concern (`config`, `state`, `frames`, `masks`, `predict`, `crop`, `propagate`,
`qc`, `orchestrator`), with `pipeline/__init__.py` re-exporting the full public surface so every caller
and test still imports `pipeline` unchanged. Pure structural move, no logic changed. Done in two steps:
`git mv pipeline.py pipeline/__init__.py` (history preserved), then one extraction commit. Submodules
use relative sibling imports leaf-first (no cycles); lazy `torch` stays inside functions;
`pipeline.config` (run knobs) is kept distinct from `sam2_utils.config` (static constants). Only the
import-direction test changed (its file glob now scans `pipeline/*.py`). Verified: 135 tests + ruff +
import-direction green; a single-chain `run_aval` smoke confirms `run_chain` behaviour end to end.
See [ADR 0013](adr/0013-pipeline-package-split.md).


---

<a id="old-9"></a>
## old §9, Raw field notes from first GUI use (pre-reorg, verbatim)

> Moved verbatim from design-notes.md §9 "Backlog from first real GUI use". These are the
> original, deliberately-unpolished field notes. The live doc reorganizes their *content* into
> a themed, re-ordered backlog (live §8) and a research-directions list (live §9), but the
> originals are preserved here exactly as written, in case the reorg dropped nuance.


Observed problems and ideas from actually driving the M4 GUI on flagged chains, not yet
scheduled, not yet decided. Recorded here so they survive the session. Four groups: **§9.1**
error detection / flagging (the M4.5-adjacent accuracy thread), **§9.2** GUI / pipeline bugs (the
M4 thread), **§9.3** research & strategic directions (bigger swings to scope later), and **§9.4**
the highest-priority meta item (repo + doc reorg). Items here are **raw field notes**, deliberately
less polished than §7/§8. (Meta-note, §9.4: this section is itself append-only and overdue for the
same reorganization it calls for.)

### 9.1 Flagging & error detection

- **Flagging is a bad A/B metric, and we've been leaning on it.** Every A/B result in §8 is
  scored on review-queue deltas, i.e. on the flag rule, which is itself unreliable. This is the
  §6 *ruler* caveat made concrete: relative deltas at fixed thresholds, never absolute
  correctness. **Implication:** the learned `P(error)` detector (§6 M4.5(a), §7 *GUI as label
  engine*) should arguably come *before* further accuracy-lever tuning, we may be optimizing
  against a noisy yardstick. Lean toward building the prediction model first.
- **Error detection is the weakest part of the system right now.** The current detector scores
  the saved chain post-hoc; it is not sampling enough of the propagation to be trustworthy.
  **Proposed minimum sampling per chain before a model exists:** the starting (anchor) frame,
  one frame in *each* direction of propagation, plus a random sample and the flagged frames.
  Sampling a single frame is too thin. **Wilder idea:** use a separate image model purely as an
  error detector (does this mask look like a plausible neuron cross-section?). Connects to §7
  *Features bound the model*, the stable-but-wrong case needs richer features than the four
  geometric/IoU signals.
- **False-positive rate is genuinely high, and the queue only surfaces flagged chains.** Even
  after the item-4 intervene-gating (§7), too many queued frames aren't real errors. Worse, the
  human only ever sees *flagged* chains, so unflagged-but-wrong chains (silent errors) are
  invisible, exactly the §7 *selection bias is the killer* failure mode. **Proposed fix: GUI
  presets.** A **verify-everything mode** (walk every frame, for data collection / the unflagged
  random sample the §7 label engine requires) plus other presets tuned for feeding the predictor
  model. The presets are how M4 collects unbiased labels, not just corrections.
- **Detection fires downstream of where the error starts.** A chain flags at frame N, but on
  rewinding the propagation error actually began a few frames upstream, it's only *detected* and
  flagged once it has degraded enough downstream. Detection must be more conservative / earlier.
  **Experiment to try:** drop the intervention threshold to a **single flag** (`qc_triage_min_signals=1`,
  the legacy "queue every flag" setting, see §7 item 4) and see how recall/precision move. This
  is the §6 M4 *strict-by-default flagging* idea, flag aggressively for recall now, loosen once
  the learned detector can set the operating point.
- **Fastest near-term win for detection is probably just more metrics.** More signals in the QC
  vector → more corroboration for `intervene` and more features for the eventual model. Cheaper
  than a model and feeds it.
- **Image-state features for the predictor model.** Beyond the geometric/IoU signals, give the
  detector *some* information about the mask relative to the image and surrounding pixels, where
  the mask sits, local intensity/texture context, boundary contrast, not just shape stats. Possibly
  partly captured by `pred_iou` already (§5 #3), but worth teasing apart: `pred_iou` is the decoder's
  own confidence, not an explicit position/context feature. A way to feed the §7 *features bound the
  model* point with richer, cheap-to-compute inputs. Tie into the §6 M4.5(a) learned `P(error)`
  detector's feature set.

### 9.2 GUI & general pipeline bugs

- **Reverse-resume starts from the wrong end (BUG).** Correcting the central/anchor frame works.
  But a *backward*-direction resume (reverse propagation, toward frame 0) appears to start from the
  **first** frame (frame 0, the far end of that direction) instead of from the corrected frame
  nearest the center. It should start at the frame closest to the anchor and propagate outward. In
  `gui.resume_propagation` the reverse branch passes `start_frame_idx=frame_idx` to
  `sess.propagate(reverse=True, …)`, so the suspect is how `PropagationSession.propagate` /
  `propagate_in_video` honor `start_frame_idx` under `reverse=True` (SAM2 reverse-tracking start
  semantics). Trace that path; this clobbers/derives the corrected side from the wrong starting frame.
- **Re-propagation loads the WHOLE chain's frames, not just the needed range (perf).** Resuming from
  frame 12 of a 44-frame chain still loads all 44 frames into the predictor (`_ensure_session` →
  `PropagationSession` → `init_state` reads the full frames_dir). Only the frames in the propagation
  direction from the start frame are actually used (~12 here). Could window the init to the needed
  range. Couples with the reverse-start bug above, both are about which frames a directional resume
  actually touches.
- **Confirmed behaviour (not a bug): re-propagation DOES re-flag.** After a resume,
  `gui._save_and_qc` re-runs `pipeline.run_qc` over the corrected masks → recomputes every QC signal,
  rewrites `qc.csv`, updates `state.json` status (flagged/done), and refreshes `triage_frames`. So
  the corrected chain is re-flagged from scratch on the new masks; no stale flags carry over. (Recorded
  per a June 2026 check.)
- **Non-central nodes don't auto-pick-up annotations.** Opening a frame whose chain node isn't
  the central one still shows "no positive nodes", the annotation isn't seeded onto the frame.
- **Painted masks sometimes change after re-propagation.** A hand-painted correction looks
  different after a resume. **Suspect:** mask post-processing (`pipeline.postprocess_mask`,
  open→close→largest-CC→fill-holes, §6 M3.5 item 5) running over the painted mask. Distinct from
  the already-fixed `correct_as_cond` silent-revert bug (§2, M4 review-testing pass item 4), worth confirming it isn't post-processing reshaping the stroke. **Action: just try scrapping
  post-processing** (it's already default-off in the headless path, so this is low-risk, only the
  save-time fold needs to skip it for painted masks) and see if the drift stops; if we keep it, **A/B
  it in the future** (post-proc on vs off, queue delta + does the painted stroke survive a resume
  bit-identical). Cheap to test, and it directly addresses a correctness complaint rather than a
  queue-size one.
- **Post-processing may be actively *hurting* results, explore alternatives.** Beyond the painted-mask
  drift above, there's a standing suspicion the current `postprocess_mask`
  (open→close→largest-CC→fill-holes, §6 M3.5 item 5) is degrading masks generally, not just painted
  ones ("a feeling the current one is absolutely ruining the results"). **Action:** survey more
  reliable post-processing approaches for thin/branching neurites (the morphological open/close kernel
  is a blunt instrument at scale-8), or confirm that *no* post-processing beats the current one. This
  is the broader version of the scrap-it experiment above: it's default-off in the headless path, so
  the first datum is just an on-vs-off A/B on real chains (queue delta + eyeball). Largest-CC in
  particular is dangerous near merges (see the branching/merging bullet below).
- **Multi-node-per-layer chains read as separate objects.** Some z-layers carry multiple nodes of
  the *same* neuron; these are read as distinct objects. In some cases the multiple nodes really
  are connected and just carry multiple annotations for viewer clarity, the GUI should recognize
  that rather than splitting them.
- **Branching / merging: the mask covers only one arm of a merge (segmentation behaviour).** Where a
  process branches or two processes merge, the propagated mask often covers only the side it was
  tracking. Picture a Venn diagram of A ∪ B: the mask captures A and A∩B but stops at the boundary,
  missing the B arm, because the prior frames' mask only covered A, so SAM2's memory biases it to
  keep tracking A. This is a fundamental propagation-memory limitation, related to (but distinct from)
  the multi-node bullet above: even when the annotation correctly marks the merge as one object, the
  mask won't grow into the unseen arm. Likely needs a seed/prompt on the B arm at the merge frame
  (a place the GUI's user-bounding-box / extra-point idea below would help), and makes largest-CC
  post-processing actively harmful here (it could delete the smaller arm). One of the harder
  correctness cases; flag for the predictor model too (a "merge frame" is a high-risk role).
- **No explicit save button for manual mask painting.** Painting persists on re-propagation, but
  there's no standalone save. A save button could double as a top-level "this mask is now
  confirmed-correct" signal (a strong positive label for the §7 label engine). May be unnecessary
  given the implicit save-on-repropagate, decide later.
- **Direction-limited resume.** Option to resume propagation in *one* direction only, for the rare
  case where the other direction is already correct (and possibly mark that direction's frames as
  confirmed-correct, more labels). Edge case; risks clobbering the correct side, needs care.
  Worth it only if the central-node-refined-but-one-side-already-good case shows up often.
- **Image contrast control.** Let the reviewer raise/lower EM contrast in the GUI.
- **MP4 generation is broken.** (Overlay video export, gif path works, mp4 does not.)
- **Regenerate GIF/MP4 after a revision.** A corrected chain's overlay GIF/MP4 is stale, re-run the
  `review.to_gif` / `to_mp4` export after a GUI resume so the artifact matches the new masks (couples
  with fixing the broken MP4 path above).
- **In-editor (napari) notifications.** Errors surface fine, but useful status messages ("no frames
  queued", "no positive point on this frame", "resume complete", etc.) currently only print to the
  terminal. Surface them in-GUI via napari's notification API so a reviewer not watching the console
  sees them.
- **User-drawn bounding box + prompts into video mode (flexibility).** Let the human draw a bounding
  box (and/or extra points) in napari and feed it to the video seed, not just the anchor box/points.
  Adds a manual lever for hard frames, e.g. seeding the B arm at a merge (see branching/merging
  bullet) or re-bounding a drifted frame. Note the §8.3 finding that box+positive is the best *AUTO*
  seed, but this is a *human* override, a different regime (like the GUI's painted-mask seed).
- **Revisit the user guide, minimize buttons.** [review-flagged-chains.md](how-to/review-flagged-chains.md) and the panel need a
  pass: fewer buttons, clearer flow. This is the §6 M4 *split marking/intervention GUI* idea
  (sweep-ok/bad mode vs fix-flagged mode) surfacing again from real use, the panel has accreted too
  many controls. Pairs with the §9.4 doc reorg.

---

### 9.3 Research & strategic directions (scope later)

Bigger swings than the §9.1/§9.2 fixes, recorded so they're not lost, not yet milestoned. Most feed
M4.5 (the predictor/model thread) or a possible M4.6+ re-architecture.

- **Working discipline (note-to-self): don't ship decisions on AI/assistant "vibes" alone, ground
  them in outside research.** The §8 levers and §9 ideas are reasoning, not evidence; before
  committing real effort, check the literature and prior art for how a problem is actually solved.
  (This applies to AI-assisted reasoning specifically: an assistant's plausible-sounding rationale is
  a hypothesis to verify, not a result.) When a claim about a method, metric, or trick comes from
  recollection rather than a read source, **mark it as to-verify**, see the FFN entry below for the
  format. This pairs with the §6/§8 *ruler* discipline (measure, don't trust).
- **Survey error-detection in (domain-specific) segmentation.** Look at how the field handles QC /
  error detection for image & video segmentation, and at pipeline-level approaches like **Seg2Track**
  and adjacent tracking-by-segmentation work, where detection/correction is built into the
  data/model pipeline rather than bolted on as post-hoc QC. May reshape the §6 M4.5(a) learned
  `P(error)` detector, or suggest moving error handling upstream into propagation itself.
- **Consult: Januszewski et al., "High-precision automated reconstruction of neurons with
  flood-filling networks" (Nature Methods 2018; bioRxiv 200675; arXiv:1611.00421; code `google/ffn`).**
  Same domain (automated EM neuron reconstruction), likely has transferable tricks. **Verified from
  the Google Research blog ("Improving Connectomics by an Order of Magnitude", 2018):**
  (i) FFN is **seed-based iterative segmentation**, seeded at one pixel, a *recurrent* CNN fills the
  object by reusing voxels already classified with high certainty from earlier iterations, moving a
  field-of-view across the volume; conceptually parallel to SAM2 propagating one object from an anchor
  with memory. (ii) Their headline quality metric is **Expected Run Length (ERL)**, "from a random
  point in a random neuron, how far can we trace before making a mistake?", i.e. error-free traced
  *distance*, which **separates merge errors from split errors**. (iii) They track **merge-rate**
  (two neurites wrongly traced as one) as its own curve. **Transferable hypotheses (TO VERIFY against
  the paper, these are recollection/inference, not yet read from source):** (a) an **ERL-style
  metric** for our chains (how many consecutive z-slices propagate before the first error) would be a
  better quality measure than per-frame flag *counts*, directly attacks §9.1 "flagging is a bad
  metric"; (b) FFN reportedly **gates field-of-view movement on the predicted mask probability at the
  FOV border**, if so, that's an *in-loop* degradation/halting signal (stop when confidence drops),
  vs our *post-hoc* QC (→ §5 #4 "QC into the loop"); (c) FFN reportedly avoids merges via
  **oversegmentation + agglomeration** and/or **multi-seed consensus** (run from 2+ seeds, flag where
  masks disagree), a consensus check maps directly onto §9.1's multi-frame sampling and the
  branching/merge failure (§9.2). Read the Methods to confirm (b)/(c) before building on them.
- **Per-frame (dense) segmentation instead of per-chain propagation.** A step back from the current
  one-object-per-chain propagation: segment *everything* in each frame, then resolve which mask is
  which object across z by sorting overlaps on score (and/or area). Could sidestep propagation drift
  and the branching/merging failure (§9.2) entirely, at the cost of a hard cross-frame association /
  overlap-resolution problem. Complex and speculative, "not fun but MUST consider." Park as a
  possible re-architecture if propagation accuracy plateaus.
- **Fine-tuning the segmentation model (M4.5(b), EM-finetuned SAM / micro_sam).** Future option,
  revisited only if measured failure rates justify it (§4 ruler). To scope ahead of time:
  (1) **data availability**, check public EM/cell segmentation sets (**MitoEM**, **LIVECell**, others)
  for suitability; (2) **refactor cost**, how much of the pipeline changes if the predictor is
  swapped/finetuned (checkpoint loading, `setup.build_predictor`, possibly mask spaces), flagged as
  a real concern; (3) **open question: does fine-tuning on still *images* improve *video*
  propagation?** (SAM2's video memory vs the image encoder, unclear the image-finetune transfers to
  tracking). Answer (3) before investing. Ties to the §7 *micro_sam* build-vs-adopt thread.

### 9.4 Highest priority: repo + documentation reorganization

**The top near-term priority (June 2026).** Both the repo layout and the docs have grown by
*appending*, never reorganizing, this section, §8, and §7 are all append-only logs, and the repo
root has accumulated scratch (A/B harnesses + their `*.log`, notebooks, one-off scripts) alongside
the real library.

- **Repo tidy.** Separate the durable library (`pipeline.py`, `sam2_utils/`, `batch.py`, `gui.py`,
  `run_aval.py`, `tests/`) from scratch/experiments (`ab_*.py` + `*.log`, `sweep_dilation.py`,
  `calibration.py`/`.ipynb`, exploratory notebooks, `somethin.txt`). Candidate: an `experiments/` or
  `scratch/` dir and an `archive/` for superseded notebooks. Decide what's reference vs deletable.
- **Docs reorg.** [README.md](../README.md) and this file (design-notes.md) need a real
  restructure, not another append: PIPELINE_CONTEXT has redundant status across §2 / §6 / §8, the
  decisions log interleaves landed + pending + rejected, and §9 is a flat dump. Consider: a crisp
  current-state summary at the top, a separate decisions/changelog, and a clean backlog, so a reader
  doesn't reconstruct status from five sections. **Caution:** this doc is the shared big-picture
  reference and is heavily cross-referenced (§-anchors are cited throughout, incl. from code comments
  and memory), a reorg must preserve or redirect those references, so it's a careful pass, not a
  quick one. **Do this before the doc grows further.**
- **Mark, don't silently move/delete.** As part of the reorg, flag anything of uncertain status for a
  closer look rather than quietly relocating or dropping it, e.g. files whose role is unclear
  (`calibration.py` is shelved-but-parked; `somethin.txt`; `datatest.ipynb`), stale `TODO`/`# [M4]`/
  `# [DEFERRED]` markers in code, doc sections that may be superseded, and any A/B harness whose
  finding is already folded into §8 (harness keepable for repro, log deletable). Tag each
  **keep / archive / delete / needs-decision** and surface the `needs-decision` set for a human call;
  the reorg's output should include that list, not just a tidier tree.

---

