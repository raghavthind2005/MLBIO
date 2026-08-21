#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-scan1
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --array=0-7
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/scan1_%A_%a.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# STAGE 1 of the full-pool scan: image + question, n=5, over ALL candidates.
# Keeps rows the image policy answers correctly (D11/D32).
#
# Runs as a SLURM job array: one shard per array task, one node (4 GPUs) each.
# Shards are independent -- no communication, no shared state, each writes its
# own scan_image_shard<i>of<n>.jsonl. A failed task can be resubmitted alone
# with --array=<i> without redoing the others.
#
# N_SHARDS MUST equal the --array width above. scan_select refuses to proceed
# unless shards 0..N-1 are all present, so a mismatch fails loudly at selection
# rather than silently producing a truncated pool.
#
# Sizing: ~27,300 rows x 5 = ~136,600 generations. Pilot 0 ran ~3,500 in 35 min,
# so ~10 min per 1,000 => ~2.9 h per node at 8 shards. 6 h gives ~2x margin.

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

N_SHARDS=8
SHARD=${SLURM_ARRAY_TASK_ID:?this script must be submitted as a job array}

POOL=${CS1_POOL_FULL:-$CS1/pool_full}
OUT=$CS1/scan
mkdir -p "$OUT"

echo "=== stage 1 (image) shard $SHARD of $N_SHARDS ==="
echo "pool: $POOL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo

python3 "$CS1/code/scan_pool.py" \
    --pool "$POOL" \
    --out "$OUT" \
    --model "$CS1_MODEL" \
    --snapshot "$CS1_SNAPSHOT" \
    --mode image \
    --shard "$SHARD" \
    --n-shards "$N_SHARDS" \
    --n-answers 5 \
    --preset untruncated \
    --max-model-len 32768 \
    --answer-max-tokens 16384 \
    --seed 0

echo
echo "STAGE 1 SHARD $SHARD COMPLETE"
