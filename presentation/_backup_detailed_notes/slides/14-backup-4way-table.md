# Backup: the 4-way comparison

<div class="mt-6 mx-auto max-w-[60rem]">

Matched on the same 605 chains (15 shared neurons). Bleed = fraction of frames with a foreign node; lower is better.

| Method | Bleed rate | Total foreign | Dropout | Coverage |
|---|---|---|---|---|
| SAM2 per-slice | 0.113 | 4,747 | 0.000 | 0.999 |
| SAM2 propagation | 0.329 | 3,425 | 0.009 | 0.871 |
| **SAM3 per-slice** | **0.087** | **757** | **0.000** | **0.999** |
| SAM3 propagation | 0.217 | 1,962 | 0.013 | 0.882 |

</div>

<div class="mt-4 mx-auto max-w-[60rem] opacity-80">

Frame rate and total severity differ: SAM3 per-slice cuts total foreign nodes 84 percent (4,747 to 757), while the affected-frame rate improves less (0.113 to 0.087).

</div>

<!--
Beats: per-slice beats propagation on every column; SAM3 beats SAM2 within each method. Quote both the rate and the severity so they are not confused.
Timing: on demand
-->
