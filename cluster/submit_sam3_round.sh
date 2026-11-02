#!/bin/bash
# submit_sam3_round.sh: queue the whole SAM3 test round in one shot.
#
# Submits Tests 2, 3, 5 and 6 (see docs/how-to/queued-sam3-narval-tests.md) with afterok
# dependencies, so every merge, score and concat step runs as its own scheduled job. The
# only thing that runs on the login node is this submit loop. Run it from the repo root
# after a git pull:
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   bash cluster/submit_sam3_round.sh
#
# Submission is manual (Duo MFA blocks a headless login); this script does not log in.
# GPU jobs use ~/sam3env (the pinned torch), CPU jobs use ~/sam2env. Every job is held to
# the 16 EXP_NEURONS by cluster/exp_neuron_chunks.txt.

set -euo pipefail

CKPT=$HOME/projects/def-mzhen/fsid/sam3_checkpoint
CHUNKS=$PWD/cluster/exp_neuron_chunks.txt
SC=/scratch/$USER

if [ ! -f "$CHUNKS" ]; then
    echo "missing $CHUNKS (the 16-neuron chunk file); regenerate with cluster/make_chunks.py" >&2
    exit 1
fi

sub() { sbatch --parsable "$@"; }   # submit and echo just the job id

# --- Test 2: membrane re-score of the two existing SAM3 trees (CPU) ----------
PS_TREE=$SC/target_perslice_only_guard_sam3_merged
T2_TREE=$SC/target_tier2_s1forced_neg_sam3_merged
j_eval_ps=$(sub --array=0-15%8 --job-name=eval_ps_sam3 --export=ALL,TREE=$PS_TREE cluster/run_eval_array.sh)
j_eval_t2=$(sub --array=0-15%8 --job-name=eval_t2_sam3 --export=ALL,TREE=$T2_TREE cluster/run_eval_array.sh)
j_cat_ps=$(sub --dependency=afterok:$j_eval_ps --job-name=cat_ps_sam3 cluster/run_py.sh python -m eval.concat_merge_shards --tree $PS_TREE)
j_cat_t2=$(sub --dependency=afterok:$j_eval_t2 --job-name=cat_t2_sam3 cluster/run_py.sh python -m eval.concat_merge_shards --tree $T2_TREE)
echo "Test 2: eval arrays $j_eval_ps $j_eval_t2 -> concat $j_cat_ps $j_cat_t2"

# --- Test 3: SAM3 config A/B, GPU seg -> merge -> score -> concat ------------
# k_max_neg in {3,0} x multimask_generous in {off,on}; two corners are existing presets,
# the k_max_neg=0 corners are the added _kneg0 presets.
presets_neg3_gen0=original_perslice_only_guard
presets_neg0_gen0=original_perslice_only_guard_kneg0
presets_neg3_gen1=original_perslice_guard
presets_neg0_gen1=original_perslice_guard_kneg0

term_jobs="$j_cat_ps:$j_cat_t2"
for v in neg3_gen0 neg0_gen0 neg3_gen1 neg0_gen1; do
    pv=presets_$v; p=${!pv}
    root=$SC/sam3ab_$v
    merged=${root}_merged
    jseg=$(sub --array=0-15%8 --job-name=sam3ab_$v \
        --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$CHUNKS,PRESET=$p,OUT_ROOT=$root \
        cluster/run_array.sh)
    jmerge=$(sub --dependency=afterok:$jseg --job-name=merge_$v \
        cluster/run_py.sh python cluster/merge_shards.py --shard-root $root --out $merged)
    jscore=$(sub --dependency=afterok:$jmerge --array=0-15%8 --job-name=eval_$v \
        --export=ALL,TREE=$merged cluster/run_eval_array.sh)
    jcat=$(sub --dependency=afterok:$jscore --job-name=cat_$v \
        cluster/run_py.sh python -m eval.concat_merge_shards --tree $merged)
    term_jobs="$term_jobs:$jcat"
    echo "Test 3 [$v -> $p]: seg $jseg -> merge $jmerge -> score $jscore -> concat $jcat"
done

# --- Test 5: full-res SAM3 (GPU, memory-heavy) -------------------------------
# The one job that can OOM. It runs on its own afterok branch; if a task dies, only the
# full-res merge is skipped, the rest of the round still completes. Scoring full-res is
# left out (very RAM-heavy); retro_eval below reports its compute + QC columns.
fr_root=$SC/sam3_fullres
fr_merged=${fr_root}_merged
j_fr_seg=$(sub --array=0-15%4 --job-name=sam3_fullres --mem=90G --time=24:00:00 \
    --export=ALL,VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$CKPT,CHUNKS=$CHUNKS,PRESET=original_fullres,OUT_ROOT=$fr_root \
    cluster/run_array.sh)
j_fr_merge=$(sub --dependency=afterok:$j_fr_seg --job-name=merge_fullres --mem=32G \
    cluster/run_py.sh python cluster/merge_shards.py --shard-root $fr_root --out $fr_merged)
echo "Test 5: fullres seg $j_fr_seg -> merge $j_fr_merge"

# --- Test 6: retro_eval over every merged tree (CPU) -------------------------
# afterany (not afterok) so a failed full-res branch cannot leave this stuck pending; it
# globs whatever _merged trees exist when it runs.
retro_dep="$term_jobs:$j_fr_merge"
j_retro=$(sub --dependency=afterany:$retro_dep --job-name=retro_sam3 --time=12:00:00 --mem=60G \
    cluster/run_py.sh python -m eval.retro_eval --membrane --glob "$SC/*_merged" --out $SC/retro_eval_sam3_round)
echo "Test 6: retro_eval $j_retro"

echo ""
echo "All queued. Watch:  squeue -u $USER"
echo "Outputs when done:"
echo "  Test 2  $PS_TREE/_merge_metric.csv , $T2_TREE/_merge_metric.csv"
echo "  Test 3  $SC/sam3ab_*_merged/_merge_metric.csv  (the negatives/generous A/B)"
echo "  Test 5  $fr_merged  (masks only; scoring skipped)"
echo "  Test 6  $SC/retro_eval_sam3_round/retro_eval.csv  (unified table)"
echo ""
echo "Note: full-res (Test 5) may OOM even at 90G. Its branch is independent, so a failure"
echo "there does not block Tests 2, 3 or 6. Re-run that array index with more --mem if needed."
