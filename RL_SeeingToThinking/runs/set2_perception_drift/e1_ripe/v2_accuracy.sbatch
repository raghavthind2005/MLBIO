#!/bin/bash
# Set-2 v2 accuracy pilot (vLLM). Usage: sbatch v2_accuracy.sbatch <smoke|full>
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/v2_%x_%j.out

set -euo pipefail
MODE="${1:-smoke}"
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} mode=$MODE $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
python3 v2_accuracy.py "$MODE"
echo "DONE $(date)"
