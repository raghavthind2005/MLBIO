#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out
# Usage:
#   sbatch --job-name=actpatch_c1 run_activation_patch.sh full 96
#   sbatch --job-name=actpatch_c2 run_activation_patch.sh llm_only 96
set -euo pipefail
COND="${1:?usage: run_activation_patch.sh <full|llm_only|vit_only> <step>}"
STEP="${2:?provide step}"
BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
DOCCI_DATA=/capstor/store/cscs/swissai/a0174/datasets/VLM-CapCurriculum-Perception-Data
JSONL=$DOCCI_DATA/perception_difficulty_curriculum.jsonl
IMGDIR=$DOCCI_DATA/images
NSAMPLE="${NSAMPLE:-300}"
CKPT=/iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints/global_step_${STEP}/actor
export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
echo "===== activation_patch (DOCCI): cond=$COND step=$STEP n=$NSAMPLE ====="
python3 "$ANALYSIS/activation_patch.py" \
    --base "$BASE" --ckpt "$CKPT" \
    --dataset docci --jsonl "$JSONL" --image-dir "$IMGDIR" --n-sample "$NSAMPLE" --seed 1 \
    --layers 8 12 16 20 24 28 32 35 --alphas 1 2 4 \
    --out "$ANALYSIS/actpatch_${COND}_${STEP}.csv"
