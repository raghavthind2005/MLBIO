#!/bin/bash
# Set3 — pre-Set-3b integrity audit (container; tokenizer for ptok cross-check; no GPU compute, no generation).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/p2audit_%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) python=$(python3 --version) $(date)"
python3 p2_audit.py
echo "DONE $(date)"
