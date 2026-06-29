#!/bin/bash
# =============================================================================
# run_weight_delta.sh — CPU-only sbatch job to compute weight deltas
#
# Loops over ALL saved checkpoints for ONE condition and appends to deltas.csv.
# Run once per condition. Results accumulate in a single CSV.
#
# Usage:
#   sbatch --job-name=wdelta_full     runs/analysis/run_weight_delta.sh full
#   sbatch --job-name=wdelta_llm     runs/analysis/run_weight_delta.sh llm_only
#   sbatch --job-name=wdelta_vit     runs/analysis/run_weight_delta.sh vit_only
#
# For probe-names only (verify classifier before real run):
#   sbatch --job-name=wdelta_probe   runs/analysis/run_weight_delta.sh full probe
# =============================================================================
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out

set -euo pipefail

COND="${1:?usage: run_weight_delta.sh <full|llm_only|vit_only> [probe]}"
PROBE="${2:-}"

BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
CKPT_ROOT=/iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
OUT=${ANALYSIS}/deltas.csv

# The analysis scripts live in EasyR1/runs/analysis on cluster scratch.
# (Copy them there from local with: scp runs/analysis/*.py cluster:$ANALYSIS/)
mkdir -p "$ANALYSIS"

export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

echo "===== weight_delta  condition=$COND ====="
echo "BASE      : $BASE"
echo "CKPT_ROOT : $CKPT_ROOT"
echo "OUT       : $OUT"

if [[ "$PROBE" == "probe" ]]; then
    echo "=== PROBE-NAMES MODE ==="
    FIRST=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sort -V | head -1)
    python3 "$ANALYSIS/weight_delta.py" \
        --base "$BASE" \
        --ckpt "$FIRST/actor" \
        --probe-names
    echo "Done. Review classifications above, then run without 'probe' argument."
    exit 0
fi

# Loop over all saved checkpoints
for CKPT_DIR in $(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sort -V); do
    STEP=$(basename "$CKPT_DIR" | grep -oE '[0-9]+')
    echo "--- step $STEP ---"
    python3 "$ANALYSIS/weight_delta.py" \
        --base "$BASE" \
        --ckpt "$CKPT_DIR/actor" \
        --condition "$COND" \
        --step "$STEP" \
        --out "$OUT"
done

echo "===== weight_delta DONE: condition=$COND ====="
echo "CSV rows in $OUT: $(wc -l < "$OUT")"
