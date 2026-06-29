#!/bin/bash
# =============================================================================
# run_mc_eval.sh — GPU job: run the MC perception probe on base or a checkpoint.
# Integration test for the probe foundation (data + model load + conv patch +
# processor + forward + letter readout). Reused later by module_graft.
#
# Usage:
#   sbatch --job-name=mc_base runs/analysis/run_mc_eval.sh                 # base model
#   sbatch --job-name=mc_c1   runs/analysis/run_mc_eval.sh full 96         # Cond1 step 96
#   sbatch --job-name=mc_c2   runs/analysis/run_mc_eval.sh llm_only 96
# =============================================================================
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out

set -euo pipefail

COND="${1:-base}"      # base | full | llm_only | vit_only
STEP="${2:-}"          # checkpoint step (omit for base)

BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
# babyVision MC probe data — copy here once from the MLBIO clone (see notes below).
DATA=/iopsstor/scratch/cscs/raghavthind/probe_data/babyvision_data

export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT_ARG=""
if [[ "$COND" != "base" && -n "$STEP" ]]; then
    CKPT_ARG="--ckpt /iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints/global_step_${STEP}/actor"
fi

echo "===== MC eval: cond=$COND step=${STEP:-(base)} ====="
python3 "$ANALYSIS/mc_eval.py" \
    --base "$BASE" \
    --data-dir "$DATA" \
    --out "$ANALYSIS/mc_${COND}${STEP:+_$STEP}.csv" \
    $CKPT_ARG
