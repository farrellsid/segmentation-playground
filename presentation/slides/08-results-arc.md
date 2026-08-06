# Per-slice beats propagation; SAM3 tightens both

<div class="mt-6 mx-auto max-w-[60rem]">

Matched on the same 605 chains (15 shared neurons). Bleed rate = fraction of frames with a
foreign node (another neuron's skeleton point) inside the mask, lower is better.

| Method | Bleed rate | Total foreign nodes | Dropout | Coverage |
|---|---|---|---|---|
| SAM2 per-slice | 0.113 | 4,747 | 0.000 | 0.999 |
| SAM2 propagation | 0.329 | 3,425 | 0.009 | 0.871 |
| **SAM3 per-slice** | **0.087** | **757** | **0.000** | **0.999** |
| SAM3 propagation | 0.217 | 1,962 | 0.013 | 0.882 |

</div>

<div class="mt-4 mx-auto max-w-[60rem] opacity-80">

Per-slice wins on every column, both backbones. SAM3 tightens both methods over SAM2, most
visibly on severity: SAM3 per-slice cuts total foreign nodes 84 percent (4,747 to 757) versus
SAM2 per-slice, a bigger jump than the frame-rate column alone suggests.

</div>

<!--
Opening: the comparison that actually matters, method x backbone, not a chronological journey

- two axes: propagation vs per-slice (method), SAM2 vs SAM3 (backbone)
- per-slice beats propagation within each backbone, every column
- SAM3 beats SAM2 within each method, but by a lot more on total severity than on frame rate,
  worth quoting both so they are not confused
- SAM3's real cost is on the next slide: it is not a clean win
- many-parameter sweeps (config A/B, the full milestone-by-milestone journey) are in the backup

Timing: 2 minutes
-->