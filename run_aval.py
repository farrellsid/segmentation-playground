"""
run_aval.py — milestone-1 bootstrap driver.

Runs ONE chain (AVAL) end-to-end through pipeline.run_chain and serializes the
resulting ChainState. This is the regression harness: its masks should match the
notebook's output for the same chain (same z-range, same mask pixels).

pipeline.py stays a pure library — this script is the only place that knows about
predictors, the CSV/CATMAID source, chains.json, and the filesystem layout.

Run it directly:  python run_aval.py
(`python pipeline.py` does nothing by design.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sam2_utils import setup, alignment, diagnostics
import pipeline
from pipeline import PipelineConfig, ChainState, run_chain, save_state


# =============================================================================
# Edit these to match your box  (mirrors the notebook's top-level knobs)
# =============================================================================
TARGET_CELL_NAME = "AVAL"
CHAIN_IDX        = 2          # notebook used cell_chain[2] (the 3rd AVAL chain)

# Data sources — the notebook's Windows paths. Point these at your local copies.
CSV_PATH    = Path(r"D:\Zhen Lab\SAM2 Segmentation\segmentation-playground\data\aggregate_data_pv.csv")
CHAINS_PATH = Path(r"D:\Zhen Lab\SAM2 Segmentation\segmentation-playground\data\chains.json")

# Outputs / scratch
OUTPUT_ROOT = Path(r"E:\ZhenLab\Data\output_masks\test2_single")
FRAMES_ROOT = Path(r"E:\ZhenLab\Data")     # SAM2 JPEG frame folders go here

cfg = PipelineConfig(
    model_size="large",
    scale=8,
    save_downscale=8,        # canonical: == scale, no resample, no 2x skeleton bug
    k_max_neg=7,
    neg_radius=150,          # accepted but unused in M1 (see build_prompts docstring)
    box_margin=10,
    output_root=OUTPUT_ROOT,
    frames_root=FRAMES_ROOT,
)


# =============================================================================
# 1. Annotations: load (CSV cache) + apply the stack->tif affine
# =============================================================================
# Swap the CSV read for a live CATMAID pull if you prefer:
#   from sam2_utils import catmaid
#   annotate_df = catmaid.fetch_all_annotations(catmaid.Catmaid())
annotate_df = pd.read_csv(CSV_PATH)

xy_tif = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
annotate_df["x_tif"] = xy_tif[:, 0]
annotate_df["y_tif"] = xy_tif[:, 1]


# =============================================================================
# 2. Chains: pick the AVAL chain
# =============================================================================
with open(CHAINS_PATH) as f:
    chains = json.load(f)

cell_chains = [c for c in chains if c["cell_name"] == TARGET_CELL_NAME]
if not cell_chains:
    raise SystemExit(f"No chains found for cell_name={TARGET_CELL_NAME!r}")
if CHAIN_IDX >= len(cell_chains):
    raise SystemExit(
        f"CHAIN_IDX={CHAIN_IDX} out of range; {TARGET_CELL_NAME} has "
        f"{len(cell_chains)} chain(s)"
    )
chain = cell_chains[CHAIN_IDX]
print(f"{TARGET_CELL_NAME}: {len(cell_chains)} chain(s); "
      f"running chain {CHAIN_IDX} with {len(chain['nodes'])} nodes")


# =============================================================================
# 3. Predictors: build image + video once, using cfg.model_size
# =============================================================================
# Note: both "large" predictors resident at once is heavy on VRAM. If you OOM,
# build the video predictor lazily after the image phase instead — but then you
# can't pass a single video_predictor into run_chain, so you'd inline the phases.
image_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="image")
diagnostics.snapshot("after image model load")

video_predictor, _ = setup.build_predictor(size=cfg.model_size, kind="video")
diagnostics.snapshot("after video model load")


# =============================================================================
# 4. Run the chain, serialize state
# =============================================================================
state = ChainState(neuron=TARGET_CELL_NAME, chain_idx=CHAIN_IDX, config=cfg)

state = run_chain(
    state,
    image_predictor=image_predictor,
    video_predictor=video_predictor,
    annotate_df=annotate_df,
    chain=chain,
    on_video_phase=diagnostics.cleanup_vram,   # reclaim VRAM between phases
)

state_path = OUTPUT_ROOT / TARGET_CELL_NAME / f"chain_{CHAIN_IDX:02d}" / "state.json"
save_state(state, state_path)

print(f"\nstatus      : {state.status}")
print(f"anchor node : {state.anchor_node_id}  (CATMAID z={state.anchor_catmaid_z})")
print(f"image score : {state.image_score}")
print(f"frames      : {state.n_frames}  (anchor frame_idx={state.anchor_frame_idx})")
print(f"masks       : {OUTPUT_ROOT / TARGET_CELL_NAME / f'chain_{CHAIN_IDX:02d}' / 'masks'}")
print(f"state.json  : {state_path}")


# =============================================================================
# 5. Final cleanup
# =============================================================================
image_predictor.reset_predictor()
diagnostics.cleanup_vram()


from sam2_utils import review
chain_dir = OUTPUT_ROOT / "AVAL" / "chain_02"

review.animate(chain_dir, preview_scale=4)          # scrubber in a notebook
review.grid(chain_dir, n=16)       # static glance, works in a script too
review.grid_flagged(chain_dir)     # only QC-flagged frames (once qc.csv exists)
review.to_gif(chain_dir, chain_dir / "aval_chain02.gif")