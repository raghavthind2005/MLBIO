#!/bin/bash
# Set3/Phase2 — A.1 splice diagnostic (processor only, no GPU work; container needed for transformers).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/p2lAd_%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) $(date)"
python3 p2_layerA_diag.py
echo "DONE $(date)"
