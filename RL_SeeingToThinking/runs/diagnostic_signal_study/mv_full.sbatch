#!/bin/bash
# Track T — FULL confirmatory generation (K draws/item/arm, recommended sampling). Usage: sbatch mv_full.sbatch
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/mvgen_%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MVDIR="${MVDIR:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv}"
export DSS="${DSS:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/dss}"
export K="${K:-5}"
export SEED="${SEED:-0}"
export GIT_SHA="$(git -C /iopsstor/scratch/cscs/raghavthind/set2_pilot/MLBIO_repo rev-parse --short HEAD 2>/dev/null || echo NA)"
cd "$DSS"
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} K=$K seed=$SEED git=$GIT_SHA $(date)"
python3 mv_gen.py full
echo "DONE $(date)"
