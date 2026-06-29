#!/bin/bash
# =============================================================================
# run_module_graft.sh — GPU job: S3 causal graft test on a trained checkpoint.
# Run AFTER mc_base validates the probe path. Works on existing Cond1/Cond2 ckpts.
#
# Usage:
#   sbatch --job-name=graft_c1 runs/analysis/run_module_graft.sh full 96
#   sbatch --job-name=graft_c2 runs/analysis/run_module_graft.sh llm_only 96
# =============================================================================
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out

set -euo pipefail

COND="${1:?usage: run_module_graft.sh <full|llm_only|vit_only> <step>}"
STEP="${2:?provide checkpoint step, e.g. 96}"

BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
DOCCI_DATA=/capstor/store/cscs/swissai/a0174/datasets/VLM-CapCurriculum-Perception-Data
JSONL=$DOCCI_DATA/perception_difficulty_curriculum.jsonl
IMGDIR=$DOCCI_DATA/images
NSAMPLE="${NSAMPLE:-300}"
CKPT=/iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints/global_step_${STEP}/actor

export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "===== module_graft (DOCCI): cond=$COND step=$STEP n=$NSAMPLE ====="
python3 "$ANALYSIS/module_graft.py" \
    --base "$BASE" \
    --ckpt "$CKPT" \
    --dataset docci --jsonl "$JSONL" --image-dir "$IMGDIR" --n-sample "$NSAMPLE" --seed 1 \
    --modes base full mlp attn late_mlp early_mlp \
    --out "$ANALYSIS/graft_${COND}_${STEP}.csv"
