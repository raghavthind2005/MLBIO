#!/bin/bash
# Set3 — complete report (secondary endpoints + descriptive). Container (frozen stats.py McNemar needs py>=3.8).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/set2_pilot/p2rep_%x_%j.out
set -euo pipefail
MODE="${1:-full}"
export PYTHONUNBUFFERED=1
cd /iopsstor/scratch/cscs/raghavthind/set2_pilot/set3
echo "host=$(hostname) python=$(python3 --version) mode=$MODE $(date)"
python3 p2_report.py "$MODE"
echo "DONE $(date)"
