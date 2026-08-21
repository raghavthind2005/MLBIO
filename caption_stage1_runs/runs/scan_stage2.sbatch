#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-scan2
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --array=0-7
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/scan2_%A_%a.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# STAGE 2 of the full-pool scan: the bare question, no image and no caption,
# n=5, over STAGE-1 SURVIVORS ONLY. Keeps rows the text alone does NOT answer.
#
# Why this stage exists: Pilot 0's no-evidence control found 34% of ViRL39K
# solvable from the question text alone, and on those rows the caption gap is
# structurally zero no matter what the caption says -- which is what diluted
# (a)-(c) to +1.7 points. Excluding them, the caption loses 12-17 points of the
# image's information.
#
# REQUIRES stage 1 to have completed AND been selected -- the survivors file is
# its output. Submit only after scan_select has run on the stage-1 shards.

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

N_SHARDS=8
SHARD=${SLURM_ARRAY_TASK_ID:?this script must be submitted as a job array}

POOL=${CS1_POOL_FULL:-$CS1/pool_full}
OUT=$CS1/scan
SURVIVORS=$OUT/survivors_stage1.json

if [ ! -f "$SURVIVORS" ]; then
    echo "ERROR: $SURVIVORS not found -- run stage 1 and scan_select first." >&2
    exit 1
fi

echo "=== stage 2 (text_only) shard $SHARD of $N_SHARDS ==="
echo "candidates: $SURVIVORS ($(python3 -c "import json;print(len(json.load(open('$SURVIVORS'))))") rows)"
echo

python3 "$CS1/code/scan_pool.py" \
    --pool "$POOL" \
    --out "$OUT" \
    --model "$CS1_MODEL" \
    --snapshot "$CS1_SNAPSHOT" \
    --mode text_only \
    --shard "$SHARD" \
    --n-shards "$N_SHARDS" \
    --candidates "$SURVIVORS" \
    --n-answers 5 \
    --preset untruncated \
    --max-model-len 32768 \
    --answer-max-tokens 16384 \
    --seed 0

echo
echo "STAGE 2 SHARD $SHARD COMPLETE"
