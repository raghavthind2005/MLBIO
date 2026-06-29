#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out
# Usage:
#   sbatch --job-name=depth_base run_depth_probe.sh                 # base model
#   sbatch --job-name=depth_c1   run_depth_probe.sh full 96         # Cond1 step 96
set -euo pipefail
COND="${1:-base}"; STEP="${2:-}"
BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
DOCCI_DATA=/capstor/store/cscs/swissai/a0174/datasets/VLM-CapCurriculum-Perception-Data
JSONL=$DOCCI_DATA/perception_difficulty_curriculum.jsonl
IMGDIR=$DOCCI_DATA/images
NSAMPLE="${NSAMPLE:-300}"
export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
CKPT_ARG=""
[[ "$COND" != "base" && -n "$STEP" ]] && CKPT_ARG="--ckpt /iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints/global_step_${STEP}/actor"
echo "===== depth_probe (DOCCI): cond=$COND step=${STEP:-(base)} n=$NSAMPLE ====="
python3 "$ANALYSIS/depth_probe.py" \
    --base "$BASE" \
    --dataset docci --jsonl "$JSONL" --image-dir "$IMGDIR" --n-sample "$NSAMPLE" --seed 1 \
    --out "$ANALYSIS/depth_${COND}${STEP:+_$STEP}.csv" \
    $CKPT_ARG
