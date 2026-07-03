#!/bin/bash
# E3/Method A — A0 mechanics pilot (smoke). Usage: sbatch --export=ALL,DSET=full,N_ITEMS=5 phaseA0_mechanics.sbatch
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/pA0_%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /iopsstor/scratch/cscs/raghavthind/set2_pilot
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} DSET=${DSET:-full} N_ITEMS=${N_ITEMS:-5} $(date)"
python3 phaseA0_mechanics.py
echo "DONE $(date)"
