# sam2_utils

Shared helpers extracted from the SAM2 test/diagnostic notebooks. Drop the
`sam2_utils/` folder next to your notebooks (or `pip install -e .` it) and
import as below.

## Layout

```
sam2_utils/
├── config.py        # paths, checkpoint registry, affine constants, CATMAID auth
├── setup.py         # device + autocast, checkpoint download, predictor build
├── viz.py           # show_mask/points/box/masks, pick_point, pick_landmark
├── diagnostics.py   # snapshot(), cleanup_vram()
├── catmaid.py       # Catmaid client + fetch_all_annotations
└── alignment.py     # fit_affine, apply_affine, catmaid_to_tif, sample_nodes_grid
```

## Quick start (Image)

```python
from sam2_utils import setup, viz, diagnostics, config
from pathlib import Path
import cv2

# 1. Device + model in one shot
predictor, device = setup.build_predictor(size="small", kind="image")
diagnostics.snapshot("after model load")

# 2. Load an image
tif_files = sorted(config.WORM_PATH.glob("*.tif"))
image = cv2.cvtColor(cv2.imread(str(tif_files[0])), cv2.COLOR_BGR2RGB)

# 3. Embed + predict
import torch, numpy as np
with torch.inference_mode():
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[4550, 4990]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

# 4. Show + clean up
order = np.argsort(scores)[::-1]
viz.show_masks(image, masks[order], scores[order],
               point_coords=np.array([[4550, 4990]]),
               input_labels=np.array([1]))
predictor.reset_predictor()
diagnostics.cleanup_vram()
```

## Interactive pickers

Run `%matplotlib widget` in a notebook cell *before* using `pick_point` or
`pick_landmark`; it's Jupyter magic™ and can't be invoked from a function.

```python
%matplotlib widget
from sam2_utils import viz, alignment

# Generic: just print clicks
fig, ax = viz.pick_point(image)

# Landmark collection for affine alignment
collected = []
target_row = annotate_1293[annotate_1293["node_id"] == 25411870].iloc[0]
fig, ax = viz.pick_landmark(image, target_row, zoom=1000, collected=collected)

# Once you have ~10-12 landmarks:
result = alignment.fit_affine(collected)
M, t = result["M"], result["t"]
```

## CATMAID

Token comes from `$CATMAID_TOKEN` or a `.env` file next to the package.
**Do not commit your token.**
Make a .env file and put in:
```
CATMAID_TOKEN=INSERTTOKENHERE
```
No spaces, no quotes!!!


```python
from sam2_utils import catmaid

cm = catmaid.Catmaid()                          # uses config defaults
print(cm.stack_info())                          # sanity-check
df = catmaid.fetch_all_annotations(cm)          # full nodes DataFrame
```

## Applying the stored affine

```python
from sam2_utils.alignment import catmaid_to_tif

xy_tif = catmaid_to_tif(df["x"].values, df["y"].values)
df["x_tif"] = xy_tif[:, 0]
df["y_tif"] = xy_tif[:, 1]
```

## Notes

- The Windows pagefile readout in `diagnostics` falls back to `psutil` on
  Linux/Mac, so `snapshot()` works across platforms.
- `setup_device()` enters a bfloat16 autocast context that stays alive for
  the session. Calling it more than once is safe — it only enters once.
- The affine constants (`M_AFFINE`, `T_AFFINE`) in `config.py` were fit to
  CATMAID z=1293. If you refit on a different section or with a larger
  landmark set, update them in `config.py` so every notebook gets the new fit.
