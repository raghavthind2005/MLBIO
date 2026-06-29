#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out
# Usage: sbatch --job-name=pdbg run_probe_debug.sh [full 96]
set -euo pipefail
COND="${1:-base}"; STEP="${2:-}"
BASE=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
DATA=/iopsstor/scratch/cscs/raghavthind/probe_data/babyvision_data
export PYTHONPATH=/iopsstor/scratch/cscs/raghavthind/code/EasyR1:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
CKPT_ARG=""
[[ "$COND" != "base" && -n "$STEP" ]] && CKPT_ARG="--ckpt /iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}/checkpoints/global_step_${STEP}/actor"
python3 "$ANALYSIS/probe_debug.py" --base "$BASE" --data-dir "$DATA" --n 6 $CKPT_ARG
