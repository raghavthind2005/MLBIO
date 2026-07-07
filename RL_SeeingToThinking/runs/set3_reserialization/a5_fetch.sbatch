#!/bin/bash
# A5 step 0 — materialize testmini from local parquet (container; pyarrow+PIL; offline).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/a5fetch_%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) python=$(python3 --version) $(date)"
python3 a5_fetch.py
echo "DONE $(date)"
