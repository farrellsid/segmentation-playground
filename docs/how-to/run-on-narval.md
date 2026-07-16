# How to run the batch in parallel on Narval

This runs the target-worm batch across many GPUs on the Digital Research Alliance of Canada
cluster Narval, one chunk of neurons per Slurm array task. It is the cluster counterpart of
[run-a-batch.md](run-a-batch.md); the design and rationale are in
[../superpowers/specs/2026-07-01-narval-parallel-batch-design.md](../superpowers/specs/2026-07-01-narval-parallel-batch-design.md).

The scripts live in `cluster/`. Only `WORM_PATH` needed a code change to run off-box: it now
reads the `SAM2_WORM_PATH` env var (see [../reference/configuration.md](../reference/configuration.md)).

## Prerequisites

- A Narval account under an allocation group (here `def-mzhen`), with an SSH key uploaded to
  CCDB and Duo MFA enrolled. Login is `ssh <user>@narval.alliancecan.ca` then a Duo prompt.
- The raw EM tif stack on project storage, e.g. `~/projects/def-mzhen/<user>/SAM2_test_NR_raw/`.
  Push it once from the local machine with `scp -r <local tif dir> narval:projects/def-mzhen/<user>/`.
- The `data/` inputs the run reads: `aggregate_data_pv.csv`, `chains.json`, `roots.json`. These are
  gitignored (`data/*`), so a fresh clone does NOT have them. Push them into the cloned repo's
  `data/` dir: `scp <local>/data/aggregate_data_pv.csv <local>/data/chains.json <local>/data/roots.json narval:projects/def-mzhen/<user>/segmentation-playground/data/`.

## One-time setup on the login node

```bash
# code
cd ~/projects/def-mzhen/<user>
git clone https://github.com/farrellsid/segmentation-playground.git
cd segmentation-playground && git checkout repo-reorg

# venv: opencv/scipy-stack/ipykernel are MODULES, load them before activating the venv
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
virtualenv --no-download ~/sam2env && source ~/sam2env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision scikit-image h5py tqdm requests   # numpy/scipy/pandas/matplotlib/Pillow/psutil come from the modules
pip install git+https://github.com/facebookresearch/sam2.git       # login node only (compute nodes are offline)

# checkpoint (compute nodes are offline, so fetch it here)
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Smoke test one neuron

Confirm CUDA and one chain end to end on an interactive GPU before launching the array:

```bash
salloc --account=def-mzhen --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=1:00:00
# on the compute node:
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
source ~/sam2env/bin/activate
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/<user>/SAM2_test_NR_raw
python batch.py --preset original --neurons AVAL --output-root /scratch/$USER/smoke --frames-root $SLURM_TMPDIR/frames
```

## Launch the full array

```bash
python cluster/make_chunks.py                 # writes cluster/neuron_chunks.txt, prints the array range
# set --array in cluster/run_array.sh to the printed 0-<N-1>%<concurrency>
sbatch cluster/run_array.sh                    # note the job id it prints
sbatch --dependency=afterok:<arrayjobid> cluster/run_merge.sh
```

Each task writes its own shard under `/scratch/$USER/target_shards/chunk_<i>/`, so parallel
tasks never share the top-level `_manifest.csv` / `_triage.csv` / `_timing.csv`. `run_merge.sh`
stitches the shards into `/scratch/$USER/target_merged/` and rebuilds the global triage queue.
Frame prep goes to node-local `$SLURM_TMPDIR`, so concurrent tasks do not contend on the
network filesystem. Sizing (`--time`, `--mem`, chunk size) starts generous; tighten it from the
first run's `_timing.csv`.

Pull the merged tree back with `rsync -L` (follow symlinks) since the merged neuron dirs are
links into the shards.

## Gotchas worth knowing

- **OpenCV is a module, not a pip wheel.** `pip install opencv-python` hits a dummy wheel that
  errors on purpose. Load `opencv/4.13.0` before activating the venv and drop it from the pip line.
- **Submit with the bare account `def-mzhen`.** `sacctmgr` lists `def-mzhen_cpu` and
  `def-mzhen_gpu`; those are the internal accounts Slurm charges. Jobs submit with `def-mzhen` and
  the scheduler routes to `_cpu` or `_gpu` based on `--gres`. Passing the suffixed name fails.
- **A brand-new account cannot submit yet.** Right after CCDB access is granted, `salloc`/`sbatch`
  fail with "No partition specified or system default partition" until the account propagates into
  the partitions' allow-lists. This is a scheduler-side sync, not a config error; wait a few hours
  and retry, or email `support@tech.alliancecan.ca`.
- **Login nodes have no GPU.** `torch.cuda.is_available()` is `False` there; test CUDA inside an
  `salloc`/`sbatch` job.
- **Module set must match the venv.** numpy, scipy, pandas, matplotlib, Pillow, and psutil are
  provided by `scipy-stack` and `ipykernel`, not the venv, so the batch scripts load the exact same
  modules before activating it.

## Resolution experiments (a separate, small comparison)

To compare ways of spending compute on resolution (design:
[../superpowers/specs/2026-07-06-fullres-resolution-experiments-design.md](../superpowers/specs/2026-07-06-fullres-resolution-experiments-design.md)),
run the experiment presets on the fixed `EXP_NEURONS` subset. These are separate from the
full target-worm run and write their own `/scratch/$USER/<preset>_*` trees. Chunks are 2
neurons each (`cluster/exp_neuron_chunks.txt`, array 0-7); submit one array per variant, then
merge each:

```bash
cd ~/projects/def-mzhen/fsid/segmentation-playground
git pull                                    # get the presets + run_exp.sh
sbatch --job-name=exp_fullres     --export=ALL,EXP_PRESET=original_fullres     cluster/run_exp.sh
sbatch --job-name=exp_wholeimg_s4 --export=ALL,EXP_PRESET=original_wholeimg_s4 cluster/run_exp.sh
sbatch --job-name=exp_tier2forced --export=ALL,EXP_PRESET=original_tier2forced cluster/run_exp.sh
sbatch --job-name=exp_bigimg      --export=ALL,EXP_PRESET=original_bigimg      cluster/run_exp.sh
sbatch --job-name=exp_tier2_s1    --export=ALL,EXP_PRESET=original_tier2_s1    cluster/run_exp.sh   # scale-1 tier-2 (the cheap resolution win)
# merge each after its array succeeds (use the job id sbatch printed):
sbatch --dependency=afterok:<jobid> --export=ALL,EXP_PRESET=original_fullres     cluster/run_merge_exp.sh
sbatch --dependency=afterok:<jobid> --export=ALL,EXP_PRESET=original_wholeimg_s4 cluster/run_merge_exp.sh
sbatch --dependency=afterok:<jobid> --export=ALL,EXP_PRESET=original_tier2forced cluster/run_merge_exp.sh
sbatch --dependency=afterok:<jobid> --export=ALL,EXP_PRESET=original_bigimg      cluster/run_merge_exp.sh
```

`original_wholeimg_s4` is the scale control: whole image like `original_fullres` but at
scale 4, so comparing the two confirms whether whole-image `scale` changes the masks (it
should not, since SAM2 resizes both to 1024).

Each merged tree carries a `_run_meta.json` (preset, knobs, git commit, actual SAM2
`image_size`) for post-hoc interpretation. `original_bigimg` (`image_size=2048`) is the
risky one: it may OOM on the 40GB A100 or fail the post-build `image_size` assertion, so an
empty shard with an OOM or assertion error in its log is expected-failure, check the log and
`_run_meta.json`.

### Per-slice reseeding variant

To test per-slice propagation with reseeding at slice boundaries, first run a downscaled
local smoke test on your GPU. `original_perslice` normally forces a full-res tier-2 pass
(`chain_crop_scale=1`), which already OOM'd once at full res, so the local smoke adds
`--no-tier2`: that keeps the run on the preset's first, downscaled `_sam` pass (`scale=8`)
and skips the full-res crop entirely. Full-res per-slice is CCDB-only, run it there instead:

```bash
py -3 batch.py --preset original_perslice --neurons AIYL --model-size tiny --no-tier2 --clean
py -3 -m eval.merge_metric --root <that run's output_root>
```

Then submit the full run on Narval:

```bash
sbatch --job-name=exp_perslice --export=ALL,EXP_PRESET=original_perslice cluster/run_exp.sh
```

The fourth comparison point is the full `original` run you already have.
