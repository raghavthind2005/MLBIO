#!/bin/bash
# Set3/Phase2 — sweep analysis (pure stdlib + frozen stats.py; container ONLY for python>=3.8 math.comb).
# No GPU compute; runs in the container because the login-node python is <3.8. Usage: sbatch p2_analyze.sbatch <smoke|full>
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/p2an_%x_%j.out
set -euo pipefail
MODE="${1:-full}"
export PYTHONUNBUFFERED=1
cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) python=$(python3 --version) mode=$MODE $(date)"
python3 p2_analyze.py "$MODE"
echo "DONE $(date)"
