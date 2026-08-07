# Method x backbone, on the node metric

<div class="mt-4 mx-auto max-w-[60rem]">

Matched on the same 605 chains (15 shared neurons). Bleed rate = fraction of frames with a
foreign node inside the mask. Dropout = mask misses its own node entirely.

| Method | Bleed rate | Total foreign nodes | Dropout |
|---|---|---|---|
| SAM2 per-slice | 0.113 | 4,747 | 0.000 |
| SAM2 propagation | 0.329 | 3,425 | 0.009 |
| **SAM3 per-slice** | **0.087** | **757** | **0.000** |
| SAM3 propagation | 0.217 | 1,962 | 0.013 |

</div>

<div class="mt-3 mx-auto max-w-[60rem] text-[0.92rem] opacity-80">

These three signals catch gross bleed and complete misses, not a mask that locked onto the wrong
compartment. A mitochondrion or a nucleus has a real, well-defined membrane too, so a mask that
captures one instead of the cell can still read clean here, that gap is the next few slides.

</div>

<!--
Opening: what the node metric alone says, method x backbone, before the honest caveat

- two axes: propagation vs per-slice (method), SAM2 vs SAM3 (backbone)
- per-slice reads better than propagation within each backbone on all three columns; SAM3 reads
  better than SAM2 within each method, more on total severity than on frame rate, worth quoting
  both so they are not confused
- deliberately not calling this "X beats Y": these three numbers are one ruler, not the whole
  truth, the next line says why
- the real caveat: none of these three numbers catch a mask that is confidently the WRONG thing,
  mitochondria and nuclei both have real membranes, so a mask stuck on one reads as clean here,
  covered honestly a few slides ahead
- SAM3's real cost is on the next slide: it is not a clean win
- many-parameter sweeps (config A/B, the full milestone-by-milestone journey) are in the backup

Timing: 2 minutes
-->