# Review masks on a Mac

For a reviewer whose job is checking and redrawing masks, on a machine with no GPU. Nothing here
needs torch, SAM2, the raw EM store, or access to the lab's drives.

## Once, to set up

```bash
git clone <repo-url>
cd segmentation-playground
python3 -m pip install -r requirements-review.txt
```

## Every session

1. Unzip the bundle you were sent, anywhere you like.
2. Run the launcher:

   ```bash
   python3 launcher.py
   ```

3. Browse to the unzipped bundle folder. The window lists its neurons and how many chains of each
   are already reviewed.
4. Tick the neurons you are working on, or leave them all unticked to get everything.
5. Leave the mode on **Redraw only**. The reprop option is greyed out unless torch is installed.
6. Click **Launch review**.

Your settings are remembered in `~/.sam2review/profile.json`, so the next session starts where you
left off.

## In the review window

| Key | Action |
|---|---|
| `,` / `.` | Previous / next flagged frame in this chain |
| `L` | Freehand lasso, adds the enclosed area to the mask |
| `Z` | Zoom to the mask |
| `S` | Save masks |
| `W` / `O` | Mark this frame wrong / ok |
| `A` / `X` | Approve / reject the whole chain |
| `Ctrl+Z` | Undo the last mask edit |

Paint and erase with napari's own brush on the `mask` layer.

The model controls (prompts, re-run, resume propagation, recrop) are not shown in redraw mode,
because they need a GPU this machine does not have.

## When you are done

Zip the bundle folder again and send it back. On the master machine, the corrections merge in with:

```bash
py -3 import_bundle.py --bundle <returned-folder> --output-root <master-tree> --dry-run
py -3 import_bundle.py --bundle <returned-folder> --output-root <master-tree>
```

Run the `--dry-run` first and read the list of chains it reports.
