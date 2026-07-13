#!/bin/bash
# Track T — generation SMOKE (mechanics only). Usage: sbatch mv_smoke.sbatch [smoke|full]
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/mvgen_%x_%j.out
set -euo pipefail
MODE="${1:-smoke}"
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MVDIR="${MVDIR:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv}"
export DSS="${DSS:-/iopsstor/scratch/cscs/raghavthind/set2_pilot/dss}"
cd "$DSS"
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} mode=$MODE $(date)"
python3 mv_gen.py "$MODE"
echo "DONE $(date)"
