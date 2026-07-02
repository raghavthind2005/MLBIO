#!/bin/bash
# Set-2 Step-0.5 pilot — Qwen3-VL-4B-Thinking chain-length gate on CLEVR.
# 1 GH200 is plenty for a 4B model. Runs fully offline (local model + local parquet).
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/pilot_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1          # model = local path, data = local parquet → no hub calls
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 pilot_chainlen.py
echo "DONE $(date)"
