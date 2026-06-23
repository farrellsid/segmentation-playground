# Review a whole neuron (`gui_neuron.py`)

The second review paradigm. Where `gui.py` opens one chain at a time, this opens a
whole **neuron**: all its chains (branches) on a single per-neuron crop canvas, shown as
one multi-color object. The branches are still tracked as separate SAM2 objects under
the hood; the neuron view is a presentation and union layer on top.

Use the per-chain `gui.py` for fine, single-branch work at full tier-2 resolution; use
this for seeing and fixing a neuron as a whole.

## Launch

```
py -3 gui_neuron.py                          # opens the picker
py -3 gui_neuron.py --neuron AVAL            # opens straight onto AVAL
```

Pick a neuron from the dock dropdown and hit **open neuron**. Opening a neuron prepares
full-resolution image windows over the neuron's whole z-range, so it is not instant; the
console prints the crop it built, for example
`[gui_neuron] AVAL: 24 branches, 312 slices, _ncrop 2048x1792px @ crop_scale 2`.

## The canvas

Everything is shown in one per-neuron crop (`_ncrop`), sized to the neuron's whole
skeleton at an adaptive resolution. A compact neuron gets a sharp canvas; a neuron whose
arbor spreads across the section gets a coarser one (the crop has to cover more area).
The canvas is never coarser than the scale-8 full frame, but for a large neuron it can
be coarser than a single branch's own tight tier-2 crop. When you need that branch at
full sharpness, open it in `gui.py` instead.

Layers:

| Layer | What it is |
|---|---|
| **neuron** | One Labels layer holding every branch. Each branch is a distinct color (its integer label is `chain_idx + 1`); every colored pixel is the neuron. The **selected label** is the **active branch** (set it from the layer's label control). |
| **EM** | The `_ncrop` image over the neuron's z-range. |
| **prompts** / **box** | Click prompts and a bounding box for the **active branch** only. |

## Correcting a branch

Corrections always act on the **active branch** (the selected label), in the `_ncrop`
canvas:

- Select the branch via the `neuron` layer's label control (or paint with that label).
- Place positive/negative points, or press `B` and drag a box.
- `R` (**re-run image phase**): re-predict the active branch on the current frame from
  its points/box; the result replaces that branch's pixels on that frame.
- `G` (**resume propagation**): re-track the active branch over the neuron frames from
  the current frame (both directions), then save just that branch. Its masks are saved in
  the `_ncrop` space and its `crop_window` becomes the neuron window.

Other branches are never touched by a correction.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `R` | re-run image phase on the active branch |
| `G` | resume propagation on the active branch |
| `B` | activate the box layer to drag a bounding box |

## Disposition

**approve NEURON** / **reject NEURON** set the review status for all of the neuron's
branches at once. A re-propagated branch is marked corrected when it saves.

## Not handled here

Cross-neuron overlap (two neurons claiming the same pixel) is not arbitrated in this
tool yet; within a neuron, branches simply union. The per-chain `gui.py` and the headless
batch are unchanged by this tool.
