# archive/ — shelved code (kept, not deleted)

Code that is **shelved but parked** — superseded or paused, kept for reference rather than deleted.
Nothing here is imported by the durable library or any driver.

| File | Why it's here |
|---|---|
| `calibration.py` | Shelved-but-parked calibration script. Imports only `sam2_utils.review` (lazily, inside a function). Its role was never folded into the durable pipeline; parked pending a keep-vs-delete call. **Needs-decision** — see [`../PIPELINE_CONTEXT.md`](../PIPELINE_CONTEXT.md) §8 item 32. |
| `calibration.ipynb` | The notebook companion to `calibration.py` (~6.8 MB). Same shelved status; pairs with the script for the delete decision. |

These are **flagged for a human keep/delete call**, not slated for deletion in the reorg pass.
