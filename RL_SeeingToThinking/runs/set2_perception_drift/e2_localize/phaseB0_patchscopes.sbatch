#!/bin/bash
# E2b Phase 0 Patchscopes go/no-go. Usage: sbatch phaseB0_patchscopes.sbatch <smoke|full>
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/pB0_%x_%j.out

set -euo pipefail
MODE="${1:-smoke}"
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} mode=$MODE $(date)"
python3 phaseB0_patchscopes.py "$MODE"
echo "DONE $(date)"
