# Backup: the many-parameter sweeps

<div class="flex justify-center mt-1">
  <img src="/images/progress-arc.png" style="max-height:130px !important; max-width:60%; object-fit:contain;" />
</div>

<div class="mt-1 text-center opacity-80 text-[0.85rem]">The full milestone-by-milestone journey: bleed rate fell from about 0.45 to about 0.09, coverage rose from about two thirds to essentially always.</div>

<div class="mt-3 mx-auto max-w-[52rem] text-[0.9rem] leading-tight">

**SAM3's negative/generous config A/B** (`perslice_only_guard` cell). Shipped preset is already
the best SAM3 setting, no retune needed.

| Config | Foreign-frame rate | Dropout |
|---|---|---|
| no negatives, not generous | 0.120 | 0.002 |
| no negatives, generous | 0.162 | 0.002 |
| **negatives, not generous (shipped)** | **0.085** | **0.001** |
| negatives, generous | 0.122 | 0.001 |

</div>

<!--
Opening: on demand, the two things behind the headline 2x2

- top: the progress arc, every milestone from the first pass to now, not just the final numbers
- bottom: negatives help on every axis, generous hurts on every axis, refuting the pre-registered
  guess that SAM3 (already more conservative) would not need negative prompts
- so the shipped preset was already the right call, this A/B just confirms it rather than
  changing anything

Timing: on demand
-->