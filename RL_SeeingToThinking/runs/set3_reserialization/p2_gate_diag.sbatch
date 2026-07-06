#!/bin/bash
# Set3/Phase2 — parity gate. Usage: sbatch p1_generate.sbatch <smoke|full>
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/p2gd_%x_%j.out

set -euo pipefail
MODE="${1:-smoke}"
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} mode=$MODE $(date)"
python3 p2_gate_diag.py "$MODE"
echo "DONE $(date)"
