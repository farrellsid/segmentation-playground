# notebooks

**Status: reference and exploration.** Not the source of truth for how the pipeline runs;
`pipeline.py` is.

Exploration / reference notebooks, kept live (not archived). They are *not* the source of truth for
*how* the pipeline runs, `pipeline.py` is, but they remain the reference for *what* each phase does
and for ad-hoc exploration.

| Notebook | Role |
|---|---|
| `single_object_depth_segmentation_.ipynb` | **Source-of-truth reference** for one chain, end-to-end (the notebook the M1 library refactor reproduces pixel-for-pixel). |
| `multi_object_segmentation.ipynb` | Multi-cell segmentation exploration (M5-adjacent). |

> `make_deck_figures.ipynb` (deck/report figure generation) was moved to
> [`../archive/`](../archive/) in the June 2026 tidy.

Run from the repo root so imports resolve (`from sam2_utils import …`).
