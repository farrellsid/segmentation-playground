# Review render: a video and a Blender mesh from a corrected bundle

**Status:** design, approved 2026-08-26
**Asked for by:** Lucinda, the second reviewer, relayed through the lab

## Why

A reviewer corrects a bundle chain by chain, one crop window at a time, and never sees the
neuron whole. Two things she cannot currently do:

1. Watch a neuron end to end to spot where the masks go wrong, rather than judging each
   chain in isolation.
2. Look at the result in 3D, where a bad slice reads as a jog in a tube and is obvious in a
   way it is not on a single cross-section.

Both are self-checks she runs on her own machine before sending work back, so the loop is
hers rather than a round trip through the lab.

## Scope

In: one video per neuron, one mesh per neuron, from either a bundle or an output tree,
driven from a window reached through `launcher.py`.

Out: anything that needs a model, anything that changes masks, per-chain outputs, and
whole-worm renders across many neurons. Those exist already in `experiments/` or are not
wanted.

## Constraints that shaped this

**She has no raw EM store and no torch.** Her install is `requirements-review.txt`. That
rules out `pipeline.load_frame_sam` on her machine and rules out anything model-shaped.

**She already has scikit-image and imageio.** `napari` declares `scikit-image[data]>=0.19.1`
and `imageio>=2.20` as hard dependencies, so marching cubes and image IO are present without
adding a line to `requirements-review.txt`. Protecting that matters, because macOS setup was
the original friction this whole review path exists to remove.

**Bundles and trees share a layout.** Both are `<neuron>/chain_NN/state.json`, so
`bundle.index_chains()` already indexes either and `pipeline.chain_masks_in_sam()` already
reads masks out of either, `_pcrop` remapped onto the `_sam` grid. No new source abstraction
is needed for chains or masks.

**They differ in exactly one way that matters here.** A bundle carries its own per-chain
`frames/`; a tree does not. That single difference is the only new seam.

## Architecture

Two files, mirroring how `launcher.py` already splits pure logic from its window so the
logic stays testable with no display.

```
render_review.py          engine + CLI + its own Qt window
  chain_frames()          THE seam: bundle frames/ vs load_frame_sam
  neuron_video()          pure: source + neuron -> gif/mp4
  neuron_mesh()           pure: source + neuron -> PLY
  render_neuron()         both, with a progress callback
  run()                   the window, thin over the above
  main()                  the CLI, thin over the same

launcher.py               gains one button that opens that window
```

`render_review.py` is standalone-runnable too, which is what serves the Windows post-merge
path; the launcher button is for her.

## The frame seam

```python
def chain_frames(chain_dir, state, source_kind):
    """{frame_idx: RGB image} for one chain, from wherever this source keeps EM."""
```

- **bundle**: read `chain_dir/frames/*.jpg` in order. No EM store, no network, works offline.
- **tree**: `pipeline.load_frame_sam(z)` for each z in `state.frame_to_z`, cropped to the
  chain's `crop_window`. Needs the EM store, which only the lab machine has.

Kind is detected once per source: `bundle.json` present means bundle, otherwise tree. A
bundle chain missing its `frames/` is an error naming the chain, not a silent skip, because a
silently skipped chain is how a video quietly stops being of the whole neuron.

## Video

One file per neuron. Chains ordered by first z, each shown in its own window, every frame
captioned with chain and z so a problem found in the video is traceable back to a chain to
re-open.

This is the tour already built for the reprop reports (`experiments/report_assets.py`), with
one change forced by the bundle case. The report version re-crops each frame out of the full
`_sam` frame, which a bundle cannot supply. So frames are **padded onto a common canvas**
instead of re-cropped.

Two consequences to handle rather than discover:

- Chains can sit at different `crop_scale`, so a raw pad would put chains at different
  magnifications in one video with nothing to signal it. Frames are normalised to a common
  nm/px first, then padded.
- The common canvas is the largest chain's normalised size, capped so one outlier chain
  cannot blow up the whole video. That cap is the same failure the merged reprop render hit
  when one chain's window dragged every frame to full-frame size.

`--format gif|mp4|both`, default `gif`. GIF is guaranteed to play; mp4 goes through cv2's
`mp4v`, which needs no extra install but does not play reliably in every browser.

## Mesh

One PLY per neuron.

1. Composite every chain of the neuron with `pipeline.chain_masks_in_sam()` onto one `_sam`
   grid, cropped to that neuron's bounding box. The crop is what keeps this laptop-sized: the
   full grid over a long neuron is hundreds of MB, the bbox is a small fraction of it.
2. `skimage.measure.marching_cubes(vol, level=0.5, spacing=...)`.
3. Taubin smoothing, **not** Laplacian. Laplacian shrinks thin tubes and a neurite is mostly
   thin tube. Implemented directly against the vertex/face arrays, no new dependency.
4. Write PLY by hand. The format is a short ASCII header plus vertex and face arrays; a
   dependency for that would not earn itself.

**Anisotropy is baked in so Blender gets real proportions.** Full-res is 16 nm in xy and
50 nm in z, and masks are scale-8, so xy is 128 nm against z at 50 nm. Note this makes **z
the finer axis**, the opposite of the usual EM intuition. `spacing=(50, 128, 128)` in
`(z, y, x)`, derived from the data rather than hardcoded, so vertices come out in nanometres.

### Detail presets

| preset | `step_size` | Taubin iters | for |
|---|---|---|---|
| `faithful` (default) | 1 | 2 | review: z-to-z jitter stays visible |
| `balanced` | 1 | 10 | general use |
| `smooth` | 2 | 25 | figures and presentations |

Default is `faithful` because the stated purpose is finding mistakes, and smoothing removes
exactly the z-to-z jitter that marks a bad slice.

**Known limitation, stated rather than implied.** Proper quadric decimation needs `trimesh`
or `open3d`, neither of which she has. `smooth` therefore reduces triangles through marching
cubes `step_size` and leans on smoothing, which is cruder than real decimation. If mesh size
becomes a problem, adding a decimation dependency is the fix, and it is a deliberate future
choice rather than an oversight here.

## The window

Reached from `launcher.py`, which already has the source picker, the neuron checklist and the
profile. The launcher passes its current path and ticked neurons across, so she picks a bundle
once.

```
Source     ~/mask-review/AIB              (from the launcher)
Neurons    [x] AIBL      [x] AIBR

Outputs    [x] Video   format ( gif  )
           [x] Mesh    detail ( faithful (for review)  )

Output to  [ AIB/review              ] [Browse]

           [ Render ]   AIBL chain_23 of 58  ----------  41%   [Cancel]
```

Presets are labelled by what they are for, not by parameter name.

**Rendering runs on a QThread.** This takes minutes, and a frozen window reads as a crash.
Progress is per chain, since the chain count is known before work starts. **Cancel stops at
the next chain boundary**, never mid-write, so a half-written PLY or a truncated video never
lands. On success the window reveals the output folder.

Settings persist through the existing `~/.sam2review/profile.json` under a `render` key, so
adding them cannot disturb what the launcher already stores.

## Errors

| case | behaviour |
|---|---|
| source is neither bundle nor tree | refuse, naming what was looked for |
| neuron has no chains with masks | skip that neuron, say so, keep going |
| bundle chain missing `frames/` | error naming the chain, do not silently drop it |
| tree source on a machine with no EM store | refuse before starting, with the reason |
| mesh volume would exceed a memory budget | refuse with the measured size and a suggestion |
| cancel pressed | finish the current chain, write nothing partial, report what was done |

## Testing

Torch-free and data-free, matching `tests/test_launcher_config.py`.

- `chain_frames` picks the right branch for a bundle vs a tree, and raises on a bundle chain
  with no `frames/`.
- Frames at differing `crop_scale` are normalised to one nm/px before padding.
- The common canvas is capped, so one outlier chain cannot size the whole video.
- Marching cubes `spacing` is derived from the data, and z is the finer axis at scale 8.
- Taubin smoothing preserves volume better than Laplacian on a synthetic thin tube, which is
  the property it was chosen for.
- PLY output parses back to the vertex and face counts that went in.
- Cancel between chains leaves no partial file.
- Preset names map to the documented `step_size` and iteration counts.

## Deliberately not doing

- **Decimation.** Needs a dependency she does not have. Named above as a future choice.
- **Per-chain videos or meshes.** ~100 files per bundle, and a neuron's chains would seam
  visibly rather than read as one neurite.
- **Both neurons in one mesh file.** One file per neuron re-exports cleanly when she
  recorrects a single neuron.
- **A napari 3D preview.** She has napari and can open a PLY in Blender; building a second
  viewer would duplicate both.
