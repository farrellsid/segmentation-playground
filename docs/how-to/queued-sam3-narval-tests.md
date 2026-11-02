# Queued SAM3 Narval tests (paste-and-go)

Four tests to run on Narval once you are back at your machine, plus the new sharded-eval
capability they use. Read [run-sam3-on-narval.md](run-sam3-on-narval.md) first for the SAM3
environment (the `~/sam3env` venv, the checkpoint upload, the torch pin) and
[run-on-narval.md](run-on-narval.md) for the general setup. This file is only the ordered
submission plan for the queued round.

Every job here is limited to the **16 EXP_NEURONS** (`sam2_utils.presets.EXP_NEURONS` =
`KEY_NEURONS` + `AVAL`): AIYR, AIYL, AIAR, AIAL, AIZL, AIZR, AIBL, AIBR, URAVR, URAVL,
URADL, URADR, RIH, RIPL, RIPR, AVAL. They are one-per-line in `cluster/exp_neuron_chunks.txt`
(16 lines), which is what pins these runs to those neurons and no others.

Two things that must not drift:

- **Submission is manual.** Duo MFA blocks a headless login, so paste these `sbatch` lines
  yourself from your terminal. Nothing here logs in for you.
- **Do not change the torch pin.** SAM3 GPU jobs use `~/sam3env` built with
  `torch==2.12.1+computecanada torchvision==0.27.1+computecanada`. A newer torch falls back to
  CPU and the run is unusably slow (see run-sam3-on-narval.md). CPU scoring jobs use `~/sam2env`
  and do not touch torch at all.

Throughout, `<user>` is your Alliance username (`fsid`), `$USER` expands on the cluster, and
`CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint` is the uploaded SAM3 checkpoint. The
account is the bare `def-mzhen`; the scheduler routes to `_gpu` when `--gres=gpu:1` is present
and to `_cpu` when it is not, so the eval jobs (no `--gres`) land on CPU nodes automatically.

## One-shot: queue the whole round

If you just want the whole round queued and walk away, run the orchestrator from the repo
root after a git pull:

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground
git pull
cat cluster/submit_sam3_round.sh   # optional: eyeball what it will submit
bash cluster/submit_sam3_round.sh
```

It submits Tests 2, 3, 5 and 6 with `afterok` dependencies, so every merge, score and
concat step runs as its own scheduled job (through `cluster/run_py.sh`) and nothing but the
submit loop touches the login node. It prints each job id and the output paths, then you
watch with `squeue -u $USER`. Test 5 (full-res) runs on an independent branch, so if it OOMs
it does not block the rest, and Test 6 uses `afterany` so it still aggregates whatever
finished. The rest of this file is the manual, test-by-test version if you want to run or
smoke them one at a time.

## The new capability: sharded, parallel scoring

`eval.merge_metric` used to score a whole tree on one core. It now takes `--neurons` (a comma-
or space-separated subset) and `--out-csv` (a per-shard destination), so `cluster/run_eval_array.sh`
can split one tree across CPU tasks the same way `run_exp.sh` splits segmentation: each array
task scores its own neuron(s) into `<tree>/_merge_metric.shard_<taskid>.csv`. After the array
finishes, `eval.concat_merge_shards` stitches those shard CSVs into the canonical
`<tree>/_merge_metric.csv` and prints the whole-tree summary. The stitched file is identical to
what a single-core run would have written.

`run_eval_array.sh` env vars: `TREE` (required, the merged tree to score), `CHUNKS` (default
`cluster/exp_neuron_chunks.txt`, the 16 neurons), `MEMBRANE` (1 = run the Phase-2 membrane pass,
the default; 0 = Phase-0 only, `--no-membrane`), `VENV` (default `~/sam2env`), `RADIUS`, `SCALE`.

If `cluster/exp_neuron_chunks.txt` is ever missing or stale, regenerate it:

```bash
py -3 cluster/make_chunks.py --chunk-size 1 \
  --neurons AIYR AIYL AIAR AIAL AIZL AIZR AIBL AIBR URAVR URAVL URADL URADR RIH RIPL RIPR AVAL \
  --out cluster/exp_neuron_chunks.txt
```

---

## Test 2 (CPU): membrane pass on the existing SAM3 trees, 16 neurons

The two SAM3 whole-set trees from the earlier run already exist on scratch. This re-scores them
**with** the membrane pass (`MEMBRANE=1`, the default) so the underfill / mild-bleed columns get
filled, restricted to the 16 neurons. No segmentation, no GPU. The trees carry more than 16
neurons (the whole-set run scored everything), so the `--neurons` filter inside the array is what
holds these numbers to the EXP_NEURONS subset.

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground

sbatch --array=0-15%8 --job-name=eval_ps_sam3 \
  --export=ALL,TREE=/scratch/$USER/target_perslice_only_guard_sam3_merged \
  cluster/run_eval_array.sh

sbatch --array=0-15%8 --job-name=eval_t2_sam3 \
  --export=ALL,TREE=/scratch/$USER/target_tier2_s1forced_neg_sam3_merged \
  cluster/run_eval_array.sh
```

When both arrays finish, stitch each (cheap, fine on the login node):

```bash
py -3 -m eval.concat_merge_shards --tree /scratch/$USER/target_perslice_only_guard_sam3_merged
py -3 -m eval.concat_merge_shards --tree /scratch/$USER/target_tier2_s1forced_neg_sam3_merged
```

Output: `<tree>/_merge_metric.csv` (16-neuron subset) plus a printed summary line with
`mild_bleed_rate`, `spanning_merge_rate`, `boundary_on_membrane`, `underfill`.

---

## Test 3 (GPU then CPU): SAM3 config A/B, 16 neurons

SAM3 is more conservative than SAM2, so negatives may hurt it. This sweeps
`k_max_neg in {0, 3}` x `multimask_generous in {False, True}` (four variants) on the per-slice +
blow-up-guard config, run with `--backend sam3`. Two corners are existing presets; the two
`k_max_neg=0` corners were added as `original_perslice_only_guard_kneg0` and
`original_perslice_guard_kneg0`. `k_max_neg=0` fully removes negatives (build_prompts caps them at
zero and propagation only seeds negatives when `k_max_neg>0`); `multimask_generous` only bites
when `multimask_anchor` is on, so the generous corners set both.

| | generous off | generous on |
|---|---|---|
| **k_max_neg=3** | `original_perslice_only_guard` | `original_perslice_guard` |
| **k_max_neg=0** | `original_perslice_only_guard_kneg0` | `original_perslice_guard_kneg0` |

Segmentation (GPU, four arrays). These reuse `run_array.sh` with `CHUNKS` pointed at the 16-neuron
file, `VENV=$HOME/sam3env`, and the SAM3 backend. Each variant gets its own `OUT_ROOT` so shards
never collide:

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground
CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint

sbatch --array=0-15%8 --job-name=sam3ab_neg3_gen0 \
  --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$PWD/cluster/exp_neuron_chunks.txt,PRESET=original_perslice_only_guard,OUT_ROOT=/scratch/$USER/sam3ab_neg3_gen0 \
  cluster/run_array.sh

sbatch --array=0-15%8 --job-name=sam3ab_neg0_gen0 \
  --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$PWD/cluster/exp_neuron_chunks.txt,PRESET=original_perslice_only_guard_kneg0,OUT_ROOT=/scratch/$USER/sam3ab_neg0_gen0 \
  cluster/run_array.sh

sbatch --array=0-15%8 --job-name=sam3ab_neg3_gen1 \
  --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$PWD/cluster/exp_neuron_chunks.txt,PRESET=original_perslice_guard,OUT_ROOT=/scratch/$USER/sam3ab_neg3_gen1 \
  cluster/run_array.sh

sbatch --array=0-15%8 --job-name=sam3ab_neg0_gen1 \
  --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$PWD/cluster/exp_neuron_chunks.txt,PRESET=original_perslice_guard_kneg0,OUT_ROOT=/scratch/$USER/sam3ab_neg0_gen1 \
  cluster/run_array.sh
```

Smoke one variant with `--array=0-0` first if you want to confirm the SAM3 env and time it;
SAM3 is roughly 3 to 4x slower per cell than SAM2, so do not trust the 12h header blindly, tighten
`--time` from the smoke shard's `_timing.csv`.

Merge each variant's shards (submit as `afterok` dependencies on the array job ids, or run once
the arrays finish):

```bash
for v in sam3ab_neg3_gen0 sam3ab_neg0_gen0 sam3ab_neg3_gen1 sam3ab_neg0_gen1; do
  py -3 cluster/merge_shards.py --shard-root /scratch/$USER/$v --out /scratch/$USER/${v}_merged
done
```

Score each merged tree (CPU, membrane on), then stitch:

```bash
for v in sam3ab_neg3_gen0 sam3ab_neg0_gen0 sam3ab_neg3_gen1 sam3ab_neg0_gen1; do
  sbatch --array=0-15%8 --job-name=eval_$v \
    --export=ALL,TREE=/scratch/$USER/${v}_merged cluster/run_eval_array.sh
done
# after they finish:
for v in sam3ab_neg3_gen0 sam3ab_neg0_gen0 sam3ab_neg3_gen1 sam3ab_neg0_gen1; do
  py -3 -m eval.concat_merge_shards --tree /scratch/$USER/${v}_merged
done
```

---

## Test 5 (GPU): full-res SAM3, 16 neurons

Segmentation with the full-res preset (`original_fullres`, scale 1, whole 9.7k-px frame) plus
`--backend sam3`. This is the memory-heavy one: SAM3 is slower than SAM2 and full-res frames are
large, so raise the memory and walltime past the `run_array.sh` header defaults on the command
line (they override the `#SBATCH` values) and keep concurrency low.

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground
CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint

sbatch --array=0-15%4 --job-name=sam3_fullres --mem=90G --time=24:00:00 \
  --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$PWD/cluster/exp_neuron_chunks.txt,PRESET=original_fullres,OUT_ROOT=/scratch/$USER/sam3_fullres \
  cluster/run_array.sh
```

Risk to watch: this can OOM even at 90G, or run long. Smoke `--array=0-0` first, watch the log
for the model load and per-chain lines, and read `peak_vram_gb` and the per-chain seconds from
that shard's `_timing.csv` before committing the whole array. If a task OOMs, its shard is
empty-or-partial (expected failure); re-run just that array index with more memory.

Merge:

```bash
py -3 cluster/merge_shards.py --shard-root /scratch/$USER/sam3_fullres \
  --out /scratch/$USER/sam3_fullres_merged
```

Scoring full-res is itself memory-heavy: the masks are scale 1 (9k x 9k), so each neuron loads a
lot of RAM. If you want per-tree merge-metric numbers, run the eval array with a larger `--mem`,
otherwise let Test 6's `retro_eval` capture its compute + QC columns and skip its merge-metric
(it skips scale < 4 by default, see below):

```bash
# optional, memory-heavy:
sbatch --array=0-15%4 --job-name=eval_sam3_fullres --mem=90G \
  --export=ALL,TREE=/scratch/$USER/sam3_fullres_merged cluster/run_eval_array.sh
py -3 -m eval.concat_merge_shards --tree /scratch/$USER/sam3_fullres_merged
```

---

## Test 6 (CPU): retro_eval across every tree, including the SAM3 ones

The unified compute + QC + merge-metric table over all merged trees. Cheap aggregation; it reads
the per-tree CSVs and (optionally) re-scores masks. Point `--glob` at the scratch trees, or list
them with repeated `--root`.

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground

py -3 -m eval.retro_eval --membrane \
  --glob "/scratch/$USER/*_merged" \
  --out /scratch/$USER/retro_eval_sam3_round
```

Two caveats baked into `retro_eval`:

- **Full-res (scale 1) is skipped for the merge-metric by default** (`--min-scale 4`) because the
  masks need a lot of RAM. Its compute (`_timing.csv`) and QC (`_manifest.csv`) columns still land
  in the table. Pass `--min-scale 1` only on a big-memory node if you want its bleed numbers too.
- **`--membrane` numbers are only comparable within one `_sam` scale.** The scale-8 trees (the A/B
  variants and the two existing SAM3 trees) compare directly; the full-res tree does not, which the
  default `--min-scale` already keeps out of the membrane columns.

Output: `retro_eval_sam3_round/retro_eval.csv`, one row per tree. Compare the SAM3 rows against the
SAM2 baselines in `docs/CHANGELOG.md`'s Phase 1 close-out and the 2-chain bake-off in
`docs/explanation/sam3-bakeoff-findings.md`.

---

## Ordered submission summary

1. **Test 2** (CPU, no dependency): two `run_eval_array.sh` arrays on the existing SAM3 trees, then concat.
2. **Test 3** (GPU): four `run_array.sh` arrays. When each finishes, `merge_shards.py`, then a `run_eval_array.sh` array per merged tree, then concat.
3. **Test 5** (GPU, memory-heavy): one `run_array.sh` array with `--mem=90G --time=24:00:00`. Merge. Score only if you raise `--mem`.
4. **Test 6** (CPU): one `retro_eval` over `/scratch/$USER/*_merged`.

Tests 2 and 6 are cheap and can bookend the round; Tests 3 and 5 are the GPU cost. Smoke one array
index of each GPU job before committing the full array, and size `--time` from the smoke shard.
