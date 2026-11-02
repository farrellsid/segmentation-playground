# Phase 2 v1 membrane map, example figures

Example outputs of the v1 classical dark-ridge membrane map
(`sam2_utils.membrane.membrane_map`, a Sato ridge filter) on real target-worm EM
frames at `_sam` scale 8, produced through `eval.merge_metric.MembraneSource`.
Sanity-check images from 2026-07-17; see the finding notes in the project memory.

## neurite-frame-membrane-map.png

A thin-neurite frame (AIZL/AIAL region). Three panels: the EM crop, the membrane
map (Sato ridge response, magma colormap), and an overlay of membrane above the
threshold in red with the mask contour in cyan. Shows the map tracing real cell
boundaries, and also firing on intracellular organelles (mitochondria, vesicle
fields), so the v1 signal is a comparative ruler, not a membrane classifier.

## blowup-frame-membrane-map-tau-sweep.png

A per-slice blow-up frame (AIZR, an 813k-px mask that engulfs the whole cross
section). Top row: EM, membrane map, overlay. Bottom row: the membrane above
three thresholds (tau 0.5, 0.65, 0.8), showing that raising tau suppresses the
organelle texture first and keeps the strong cell boundaries and outer cuticle
longest. Useful for illustrating both the blow-up failure mode and the threshold
as a specificity knob.

## Regenerating

The generating scripts live in the session scratchpad (throwaway). To reproduce,
point a short script at a merged run tree, build `MembraneSource(scale=8)`, pick a
mask via `pipeline.chain_masks_in_sam`, and call `map_for` plus the detectors.
