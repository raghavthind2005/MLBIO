#!/bin/bash
#SBATCH --job-name=ca21_t0c
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0c_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0c_%j.out

# T0c -- the first full training smoke. 2 steps, Arm B.
#
# NOT a result. What it proves is that rollout -> reward -> advantage -> caption term ->
# composed batch -> update_policy survives real shapes. Every component has been tested
# alone; none of them have ever run together.
#
# 4 GPUs on one node: world_size=4, and g_c=8 is a multiple of it, so the composed
# answer+caption batch divides evenly on every step regardless of how many prompts survive
# the correctness gate (main_ca21 checks this before allocating a GPU).
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "[job] config: $CA21/configs/ca21_smoke.yaml"

srun --environment="$CA21_TOML" \
    python3 "$CA21/code/main_ca21.py" \
        config="$CA21/configs/ca21_smoke.yaml" \
        trainer.experiment_name="t0c_smoke_${SLURM_JOB_ID}"

echo "[job] done at $(date -Is)"
