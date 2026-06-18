# finetune

**Status: scaffold, not yet implemented.** A home for future SAM2 finetuning. See the
[roadmap](../docs/explanation/roadmap.md).

**This is an empty home for the Stage 2 work. No finetuning code lives here yet**, only the
`finetune.py` placeholder stub carried over from the old `finetuning/` dir.

Per [`../docs/explanation/roadmap.md`](../docs/explanation/roadmap.md) §5 **Stage 2** and §4.7. Stage 2 is gated on
Stage 0 (the [`../eval/`](../eval/) ruler must exist first) and runs after the Stage 1 free wins.

## What goes here (when built)

Finetune SAM2 on the confirmed cross-worm ground truth:

- **Decoder-only first** (frozen prompt encoder); add **LoRA on the image encoder** if needed, PEFT updates <5% of params and fits a single consumer GPU (the explicit low-data recommendation).
- **Supervise only on confirmed voxels.**
- **Composite loss:** BCE + soft-Dice + **soft-clDice** (the topology-preserving centerline loss for
  thin neurites; FUTURE_DIRECTIONS §4.4).

Tooling/sources: micro_sam (+ peft-sam), lightweight SAM2 microscopy finetuning, SAM2LoRA, FGNet;
optionally initialize from CEM500K EM-pretrained features. Full citations in
[`../docs/explanation/roadmap.md`](../docs/explanation/roadmap.md) §4.7 and §8.

**Advance gate (Stage 2 → Stage 3):** the finetuned model beats stock SAM2 on held-out confirmed
segments. **Pivot if not:** inspect the domain gap and lean on Stage 3 (dense + cross-z linking).

> Open question to resolve here (§4.7): does finetuning on still *images* improve *video*
> propagation? SAM2's video memory vs the image encoder is the crux.

Training input GT lands in [`../data/groundtruth/`](../data/groundtruth/).
