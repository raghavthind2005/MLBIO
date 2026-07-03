#!/bin/bash
# E3/Method A — instrumented HF sweep. Usage: sbatch --export=ALL[,N_RIPE=..,MAX_NEW=..] phaseA2_hf.sbatch <smoke|full>
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/pA2hf_%x_%j.out

set -euo pipefail
MODE="${1:-smoke}"
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} mode=$MODE $(date)"
python3 phaseA2_hf.py "$MODE"
echo "DONE $(date)"
