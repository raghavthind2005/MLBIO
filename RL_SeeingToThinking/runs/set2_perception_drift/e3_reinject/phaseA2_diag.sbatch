#!/bin/bash
# E3/Method A — A2 injection-procedure diagnostic (HF, output_attentions). Usage: sbatch --export=ALL[,N_ITEMS=..] phaseA2_diag.sbatch
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/pA2d_%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} N_ITEMS=${N_ITEMS:-5} $(date)"
python3 phaseA2_diag.py
echo "DONE $(date)"
