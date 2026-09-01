# Set up SAM2 on a review machine

For a reviewer who has been redrawing masks by hand and now needs **recrop** or the
re-segmentation keys. The plain review install deliberately has none of this, so nothing
here is required until you want those controls.

Three things have to line up: torch, a checkpoint, and the raw EM. The launcher's
**Check this machine** button reports all three, and this page is what to do about each.

## First, what actually needs the model

| Control | Needs torch + checkpoint | Needs raw EM |
|---|---|---|
| Painting, lasso, save, approve, reject | no | no |
| `R` re-run image phase | yes | no |
| `G` resume propagation | yes | no |
| `C` / `F` recrop | yes | yes |

**Recrop is a full re-run, not a crop.** It cuts the new window out of the raw EM, then
re-seeds the anchor and re-propagates the whole chain through SAM2. That is why it needs
everything in the table, and why it is the slowest thing in the GUI. If you were
expecting it to just move the viewing window, it does more than that.

## 1. Install torch into the interpreter that runs the launcher

This is the step that usually goes wrong, and it fails in a confusing way: torch is
installed, but the launcher says it cannot import it. That means it went into a
different Python than the one starting the launcher. Your review environment was
created without torch on purpose, so a torch installed afterwards has to land in that
same environment.

Check which interpreter you are actually using. Run this in the shell you start the
launcher from:

```bash
source ~/review-env/bin/activate     # the environment from the original setup
python3 -c "import sys; print(sys.executable)"
python3 -c "import torch; print(torch.__version__)"
```

If the first prints something under `~/review-env` and the second prints a version, you
are done with this step. If the second says `No module named 'torch'`, install it there:

```bash
source ~/review-env/bin/activate
python3 -m pip install -r ~/segmentation-playground/requirements.txt
```

That pulls a few GB, unlike the review requirements file. It is the one time to let it
run.

The launcher names the interpreter it is using, so if it still reports torch as not
importable, compare the path it prints against the one from `sys.executable` above. Two
different paths means the launcher is being started from outside the environment.

## 2. Get the checkpoint

The model weights are one file:

- name: `sam2.1_hiera_large.pt`
- size: about 860 MB
- source: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

Two ways to get it, and the first needs no action:

**Let it download.** Point the launcher's **Checkpoints** field at any folder you can
write to, for example `~/sam2-checkpoints`. The first time you press `R`, `G` or `C`,
the file downloads into that folder. It happens once.

**Or copy it across.** If the download is slow or you have it already, put the file in
that folder yourself, keeping the name exactly as above. The launcher looks for that
filename and nothing else.

An empty folder is fine as a starting point, but an empty folder is not a substitute for
the weights: the download still has to happen before a recrop can run.

## 3. Point at the raw EM

Recrop reads the full-resolution tif stack, the files named like `1301____z1300.0.tif`.
Set **Raw EM (tif stack)** to the folder holding them. The check rejects a folder of
unrelated tifs, so if it says the folder holds no `z<number>.tif` frames, you have
pointed at the wrong place.

Nothing except recrop needs this, so a machine without the raw EM can still redraw,
re-predict and re-propagate.

## 4. Confirm before you start a session

In the launcher, press **Check this machine**. You want five lines, and `ok` on torch and
checkpoint is what unlocks the model modes:

```
ok  torch: version 2.x
ok  device: mps (Apple Silicon) ...
ok  checkpoint: sam2.1_hiera_large.pt present in ~/sam2-checkpoints
ok  raw EM: tif stack at ...
ok  frames cache: writable at ...
```

A missing checkpoint still reports `ok`, with a note that it will download on first use.
That is deliberate: it warns you about the wait rather than blocking the session.

## What to expect on a Mac

The device line will say `mps`. Two things follow from that, and neither is a fault:

- It is much slower than the lab GPU. A recrop re-propagates a whole chain, so budget
  minutes per chain, not seconds. The window is blocked while it runs.
- SAM2 on MPS is still called preliminary by the people who wrote it, so a mask can come
  out slightly different from what the same click produces on the lab machine. If a
  result looks odd, that is worth knowing before you chase it.

## Modes

The launcher's **Mode** dropdown has three settings:

- **Redraw only**, no model at all. This is the default and needs none of this page.
- **Recrop only**, which offers `C` and `F` without the prompt controls. It is not gated
  on torch, so it will let you start a session and then fail when the recrop actually
  runs. Use it once the setup above is done.
- **Enable SAM2/SAM3 reprop**, everything. Stays disabled until torch and a usable
  checkpoint folder are both present.

## When something goes wrong

| What you see | What it means |
|---|---|
| `torch: not importable by /some/path` | Torch is missing from *that* interpreter. Compare it with `sys.executable`, see step 1. |
| `torch: installed, but it failed to load: ...` | Torch is there but broken in this process. Start from `launcher.py` rather than importing things yourself; it loads torch before Qt, which some machines require. |
| `raw EM: ... holds no ..z<number>.tif frames` | Wrong folder, see step 3. |
| A crash the moment you press `C` | Almost always the model, not the crop: recrop builds both predictors before it does anything. Check torch and checkpoint first. |
| `checkpoint: ... cannot be created` | The Checkpoints folder is not writable. Pick another. |

If none of those match, copy the message out of the terminal and send it on. Send the
exact text rather than a summary of it, since the messages are written to name the
cause.
