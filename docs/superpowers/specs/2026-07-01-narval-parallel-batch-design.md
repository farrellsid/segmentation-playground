# Design: parallel target-worm batch on Narval

Date: 2026-07-01
Status: proposed

## Goal

Run the target-worm `batch.py` (preset `original`, ~135 neurons in `ALL_NEURONS`) across
many GPUs on the Digital Research Alliance of Canada cluster Narval, instead of one neuron
at a time on a local machine. Split the work into parallel Slurm array tasks, then stitch
the per-task outputs back into one manifest and triage queue.

## Context

`batch.py` is already close to ready for this:

- It accepts `--neurons` (an allow-list), `--output-root`, and `--frames-root`.
- On-disk outputs are per-neuron isolated: `<output_root>/<neuron>/chain_<idx>/`.
- The only shared state per `output_root` is three top-level CSVs: `_manifest.csv`,
  `_timing.csv`, `_triage.csv`.

So the parallel run is mostly orchestration around `batch.py`, not a rewrite of it. Each
array task runs `batch.py --neurons <chunk>` into its own output shard. A merge step at the
end rebuilds the global ledger, which `build_triage_queue` already does by reading each
chain's on-disk `qc.csv`.

Cluster facts (from CCDB):

- User `fsid`, allocation group `def-mzhen` (owner Mei Zhen), 1 TB project storage.
- Project path: `~/projects/def-mzhen/fsid/`.
- Access is opportunistic (no RAC), so jobs run at default priority.
- Login uses an ed25519 key plus Duo MFA. Compute nodes have no internet.

Data:

- Target-worm EM stack: 338 `.tif` files, 26.8 GB, flat directory, named `..z####.0.tif`,
  read by `pipeline.frames.TifFrameStore` via a `*.tif` glob on `config.WORM_PATH`.
- Small in-repo inputs: `data/aggregate_data_pv.csv`, `data/chains.json`.
- SAM2 `large` checkpoint (~900 MB) downloads from a Facebook URL; must be fetched on the
  login node since compute nodes are offline.

## Granularity: chunk of neurons per task

One array task processes a chunk of ~7 neurons (about 20 tasks over ~135 neurons). This
amortizes one SAM2 model load across several neurons per task and evens out runtimes,
versus one task per neuron (135 model loads, uneven lengths) or one task per chain (a model
load per chain, overhead dominates short chains).

## Collision handling

Parallel tasks must not write the same `output_root`, or they clobber each other's ledger
CSVs. Each task writes to its own shard `output_root` under scratch:
`$SCRATCH/target_shards/chunk_<i>/`. The per-neuron dirs live inside the shard, naturally
isolated. A dependency job merges the shards after the array finishes.

## Components

Only one change to existing code; everything else is new files under a `cluster/` dir kept
out of the library (so it does not break the import-direction rule).

1. `sam2_utils/config.py`: make `WORM_PATH` environment-overridable. `OUTPUT_ROOT` and
   `FRAMES_ROOT` already have CLI flags; `WORM_PATH` does not, so it is the one path a
   cluster run cannot redirect today.

   ```python
   WORM_PATH = Path(os.environ.get("SAM2_WORM_PATH", r"E:\ZhenLab\Data\SAM2_test_NR_raw"))
   ```

   Windows behaviour is unchanged when the variable is unset.

2. `cluster/make_chunks.py`: read `ALL_NEURONS`, write `neuron_chunks.txt`, one line per
   chunk, space-separated neuron names. Chunk size is a CLI argument (default 7).

3. `cluster/run_array.sh`: the `sbatch` array script. Reads its chunk line, exports
   `SAM2_WORM_PATH`, and runs `batch.py` into a per-task shard, with frame prep on
   node-local `$SLURM_TMPDIR`.

4. `cluster/merge_shards.py`: run as an `afterok` dependency. Link each shard's `<neuron>/`
   dirs into one merged tree, concatenate the shard `_manifest.csv` and `_timing.csv`, and
   call `batch.build_triage_queue` to regenerate the global `_triage.csv`.

## Data flow

- Raw tif stack lives on project storage (persistent, backed up):
  `~/projects/def-mzhen/fsid/SAM2_test_NR_raw/`, pointed to by `SAM2_WORM_PATH`.
- Each task derives its scaled JPEG frame cache on node-local `$SLURM_TMPDIR` (fast, private
  to the task, auto-cleaned), so concurrent tasks do not contend on the network filesystem.
- Outputs are written to per-task shards on scratch, then merged.
- The merged output tree is what gets scored or pulled back for review.

## Environment (login node, once)

```
module load python cuda cudnn
virtualenv --no-download ~/sam2env && source ~/sam2env/bin/activate
pip install --no-index torch torchvision numpy scipy scikit-image pandas matplotlib \
    opencv-python Pillow h5py tqdm psutil
pip install git+https://github.com/facebookresearch/sam2.git   # login node only, needs internet
```

The GUI dependencies (`napari`, `PyQt5`, `magicgui`, `dask-image`) are not installed: they
are only needed for the local review tool, not the headless run.

## Resources (start generous, tune later)

Placeholder sbatch header, sized within Narval's per-GPU ratio (4 GPUs, 48 cores, ~498 GB
per node, so up to 12 cores and ~124 GB per GPU):

```
#SBATCH --array=0-19%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
```

The first run's `_timing.csv` gives real per-chain numbers to tighten `--time`, `--mem`, and
chunk size on later runs.

## Testing before the full run

1. Smoke test in an interactive GPU session (`salloc --gres=gpu:1 --time=1:00:00 ...`): run
   `batch.py --preset original --neurons AVAL --output-root $SCRATCH/smoke --frames-root
   $SLURM_TMPDIR/frames` and confirm masks, `qc.csv`, and the manifest look right.
2. Confirm `torch.cuda.is_available()` and the checkpoint load in that session.
3. Run the array on a 2-chunk subset before scaling to all chunks.

## Open items

- Exact per-chain runtime is unknown until the first run; the resource header is a generous
  guess until then.
- Pulling merged outputs back to the local machine (or keeping them on scratch for scoring)
  is decided after the first run produces results.
