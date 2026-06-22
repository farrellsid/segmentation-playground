# SAM2 review GUI, user guide (Milestone 4)

The napari **review / triage / correction** tool (`gui.py`). It opens chains the
headless batch flagged, lets you inspect *why* they flagged, fix the bad ones, and
records what you decided. Everything you do here is also captured as training data
for the future learned QC model (M4.5).

> **It's a post-batch tool.** `batch.py` already ran every chain end-to-end (anchor →
> propagate → save → QC). This GUI opens the chains QC *flagged* so you can correct
> them. Propagation already happened, you're reviewing and re-propagating its output,
> not driving the first pass frame-by-frame.

---

## 1. Launch

```bash
pip install napari magicgui            # one-time, GUI-only deps
py -3 gui.py                           # opens on the first chain in the queue
py -3 gui.py --neuron AIAL --chain 0   # open a specific chain
py -3 gui.py --reviewer sf             # stamp your name on the labels you create
```

Other flags: `--output-root <dir>` (which output tree to review), `--point-size N`,
`--no-auto-zoom`, `--hires-em` (full-res EM background, see §6). From a notebook:

```python
from gui import launch
launch(neuron="AIAL", chain_idx=0, reviewer="sf")
```

The first time you trigger a re-segmentation (`R`/`G`), the SAM2 models load (a few
seconds + GPU memory). Everything before that, browsing, scrubbing, labeling,
painting, needs **no GPU**.

---

## 2. The window

A frame slider (top/bottom) scrubs through the chain by **frame index**. Layers
(left list), top to bottom:

| Layer | What it is | Editable? |
|---|---|---|
| **box** | A blue bounding box, an optional seed for the image-phase re-predict (see §5b). One per frame; pre-loaded with the chain's *original* box at the anchor frame. | **Yes**, press `B` (or "draw box") then drag; select+delete to remove |
| **prompts** | The SAM2 seed points: **green = positive**, **red = negative**. Pre-loaded with the chain's *original* seed at the anchor frame. | **Yes**, click to add (in *add* mode), select+delete to remove |
| **skeleton** | Yellow dots = this chain's CATMAID skeleton nodes per slice. Context only, these are *not* the SAM prompts. | No |
| **mask** | The segmentation, as a paintable label layer. **Paint here** to correct a mask by hand. | **Yes**, napari brush/eraser; `Ctrl+Z` undoes |
| **EM** | The electron-microscopy image. | No |

The dock panel on the right is grouped: **chains · frames (this chain) ·
prompts · view · correct · label / disposition**.

---

## 3. "Next CHAIN" vs "next flagged FRAME"

These move at two different scales:

- **next flagged FRAME** (`.`) / **prev flagged FRAME** (`,`), stay *in the current
  chain*, jump to the next/previous frame QC queued. Use these to walk the problem
  spots of the chain you're reviewing.
- **prev / next CHAIN** (buttons), *close this chain and open another*. They **cycle**
  the same list the picker shows (see §3a), and wrap around. In **flagged only** mode
  that is every chain still needing a human, **including ones you've opened but not yet
  finished** (status `in_review`), so you can always come back to an unfinished chain; a
  chain leaves the cycle once it's **approved / rejected / corrected**. In **everything**
  mode it cycles every chain on disk.

So: frames are *within* a chain; the picker list is *across* chains.

## 3a. Picking a chain: flagged only vs everything

The **chains** group has a mode toggle (`show`) and two cascading dropdowns: pick a
**neuron**, then a **chain**, then hit **open selected chain**.

- **flagged only** (default): the dropdowns and the prev/next CHAIN cycle list only
  the chains QC flagged and you haven't dispositioned yet, the review queue. This is
  the post-batch triage workflow.
- **everything**: they list *every* chain saved on disk, flagged or not, so you can
  open and proofread any chain during manual review. Each chain in the **chain**
  dropdown carries a status badge, e.g. `chain_03 [flagged]`, `[done]`, `[approved]`,
  `[corrected]`, so you can see at a glance which need attention. The badge is your
  review disposition when there is one, otherwise the batch's execution status.

The mode is the single source of truth: it drives the dropdowns **and** the prev/next
CHAIN cycle, and `↻ refresh queue` re-reads from disk in whichever mode you're in.

---

## 4. Keyboard shortcuts

| Key | Action |
|---|---|
| `.` / `,` | next / prev flagged **frame** (this chain) |
| `p` / `n` | set new prompt points to **p**ositive / **n**egative |
| `B` | **draw box**, activate the box layer to drag a bounding box on this frame |
| `R` | **re-run image phase**, re-predict the anchor mask from the current points and/or box |
| `G` | **resume propagation**, re-track from the current frame over the correction |
| `W` / `O` | mark the current **frame** **w**rong / **o**k (uses the error-type picker) |
| `A` / `X` | **approve** / **reject** the whole **chain** |
| `Z` | zoom the camera back onto the mask |

(All also have buttons in the dock.)

---

## 5. Workflows

### 5a. Approve a chain that's actually fine
Many flags are benign (e.g. a skeleton node briefly leaves the plane). Scrub the
flagged frames (`.`), and if the masks look correct, hit **approve** (`A`). This
keeps the masks as-is, marks the chain `approved`, and logs the frames as `ok`.

**How seeding works (read once):** propagation is always seeded with the **mask** on
the current frame, the re-predicted and/or hand-painted mask in the **mask** layer, *not* a bounding box. `R` turns your points (and an optional drawn box, §5b) into a mask
you can preview and tweak; `G` takes whatever mask is on the current frame and
propagates it. Direction is **away from the anchor**, so an already-good segment is
never re-tracked:

- correcting the **anchor** frame → propagates **both ways** (re-does the whole chain);
- a frame **after** the anchor → **forward only** (anchor → here is preserved);
- a frame **before** the anchor → **reverse only** (here → anchor is preserved).

### 5b. Fix a bad anchor, then re-propagate
1. Go to the anchor frame (you open on it). The original seed points are already there.
2. Edit them: click to add positives (`p`) / negatives (`n`) on the neurite vs its
   neighbours; select and delete bad points. (**Reset prompts** restores the original
   seed, §5e.)
3. **Re-run image phase** (`R`), re-predicts the anchor mask from your points into the
   **mask** layer. Tweak it by painting if needed.
   - *Optional box.* If points alone won't capture the neurite's full extent, press
     `B` (or "draw box") and drag a box around it, then `R`. The box and any points go
     into SAM2 together (box-only works too). The box shapes only this image-phase
     mask; `G` still propagates the resulting **mask**, never the box. The saved box is
     pre-loaded at the anchor, so you can nudge it instead of drawing from scratch.
4. **Resume propagation** (`G`), re-tracks the whole chain from the re-seeded anchor
   (anchor → both directions).
5. Inspect, then **approve** (`A`). The corrected `masks/` + `qc.csv` + `state.json`
   are rewritten on disk, identical in form to a fresh batch run.

### 5c. Fix a mid-propagation drift
1. Scrub to the first frame where it goes wrong (`.`).
2. Either **paint** the correct mask into the **mask** layer (brush/eraser), or place
   points and `R` to make a mask there.
3. **Resume propagation** (`G`), re-tracks **only the drifted tail** (the direction
   away from the anchor), seeding with your mask. Frames between the anchor and here
   keep their existing masks, so a center frame you already corrected isn't clobbered.
4. Approve when satisfied.

### 5d. Reject a chain you can't fix
If a chain is hopeless (e.g. the anchor is on the wrong object), pick the **error
type** from the dropdown and hit **reject** (`X`). This marks the chain `rejected`
and logs its frames as `wrong` with that error type. **It does not delete the masks**, it just records the verdict; redo or exclude the chain downstream.

### 5e. Reset prompts
**⟲ reset prompts to original** discards your point edits and restores the chain's
original saved seed at the anchor frame. (Prompt-only, it does not undo mask paints;
use `Ctrl+Z` on the mask layer for those.)

### 5f. Label while you scrub (helps the M4.5 model)
While walking frames, **mark FRAME wrong** (`W`) / **mark FRAME ok** (`O`) records a
per-frame verdict for the current frame using the selected error type. Marking a
*non-flagged* frame wrong is especially valuable, it's a "silent error" the rule
missed, which the queue alone can never surface.

---

## 6. Why it looks low-res (and `--hires-em`)

For a **standard (`_sam`) chain** the mask and EM frames display at **scale-8**
(~1152×1154 from a ~9216×9230 EM frame), because that is the resolution the pipeline
*propagated and saved* it at. The high-res anchor crop only sharpens the one-frame seed;
its box is mapped back to scale-8 and the crop is discarded.

For a **tier-2 chain** (run with `chain_crop=True`) the mask and frames display at the
chain's **crop resolution** (`_pcrop`, typically scale-2), genuinely sharper, because the
whole chain was propagated and saved in that crop. The GUI detects this from the chain's
`state.json` (`crop_window`) and rebuilds the crop space automatically; nothing to toggle.

- For an `_sam` chain the **mask cannot be sharpened in the GUI**, genuinely higher-res
  masks require re-propagating at higher resolution, i.e. re-running that chain with
  `chain_crop=True` (the tier-2 per-chain crop).
- `--hires-em` loads the **full-resolution EM** as the background (lazy) and scales
  the still-scale-8 mask/points to overlay it, so the EM context is crisp for judging
  a fix. The mask stays blocky, that's expected.

Note: the GUI's re-predict (`R`) uses the legacy full-frame scale-8 path (it matches
the displayed frame), so it won't reproduce the batch's high-res-crop anchor
pixel-for-pixel, it'll be a bit coarser. Crop-space re-predict is a deferred item.

---

## 7. What gets written to disk

All under your `--output-root`:

- `<neuron>/chain_NN/masks/`, `qc.csv`, `state.json`, **rewritten** only when you
  **resume propagation** (a correction). Approve/reject/label never touch the masks.
- `_review.csv`, the GUI's review-status ledger (`unreviewed → in_review →
  approved/rejected/corrected`), **separate** from the batch's `_manifest.csv`.
- `_labels.csv`, one row per labelled frame: the QC signal vector + your verdict +
  error type + the chain's anchor verdict + the frame's role. This is the M4.5
  training set. Approve/reject log the queued frames in bulk (+ a random sample of
  un-flagged frames); `W`/`O` log single frames.

### Error types
`wrong_object` (segmented the wrong neurite) · `under` (mask too small / misses part)
· `over` (mask too big / grabs background) · `bleed` (leaks into a neighbour) ·
`fragmented` (broken into pieces) · `missing` (no/empty mask) · `other`.

---

## 8. Current limitations & planned work

Known limits (deferred):

- **Mask resolution is scale-8 for `_sam` chains** (§6), working with those masks is
  "pixel art"; you often can't see the true boundary. The fix is re-propagating that chain
  at higher resolution via the **tier-2 per-chain crop** (`chain_crop=True`), now landed, tier-2 chains open at crop resolution. For an `_sam` chain `--hires-em` still only
  sharpens the EM background; re-run the chain with `chain_crop=True` for a sharper mask.
- **Re-predict matches the displayed frame**: scale-8 full-frame for an `_sam` chain (so it
  won't reproduce the batch's anchor crop pixel-for-pixel); for a tier-2 chain it runs in
  `_pcrop`, i.e. at the chain's crop resolution.
- **No concurrent reviewers**, `_review.csv` has no cross-process lock yet; one
  reviewer at a time. "↻ refresh queue" re-reads the manifest on demand (e.g. to pick
  up chains a still-running batch just flagged).
- **The model isn't trained yet**, this milestone *collects* labels; training the
  learned QC detector on them is M4.5.

Planned changes from review-testing feedback (tracked in `design-notes.md` §7):

- **Marking/intervention split (the too-many-buttons fix).** A two-mode flow: a
  *marking* mode that loads a chain and lets you sweep every frame ok/bad, and a
  separate *intervention* mode that shows only the frame(s) you flagged for fixing.
  Reduces the dense single panel and stops accidental edits while scrubbing.
  *(Planned, likely alongside M4.5; the current single panel works in the meantime.)*
- **Strict-by-default flagging.** Tune the QC thresholds to flag *aggressively* now
  (high recall, catch everything, accept false alarms) and loosen later once the
  M4.5 learned detector has labels to set the operating point. *(A pipeline-config
  change + re-run; see §7. Recommended values are noted there.)*
- **Higher-res masks (tier-2 crop).** See above, M4.5.

See `design-notes.md` §6/§7 for the full roadmap and rationale.
