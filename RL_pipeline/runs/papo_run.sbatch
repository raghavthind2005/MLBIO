#!/bin/bash
# PAPO run B (perception loss ON) — submit with `sbatch papo_run.sbatch`.
# Detached; runs the canonical papo_run.sh (max_steps=60, save_freq=6 -> 10
# checkpoints, gpu_memory_utilization=0.85, use_kl_prcp=true). ~60 steps ≈ 8-10h;
# 12h budget leaves margin (resumable from $RUN_DIR/checkpoints if it doesn't finish).
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_B
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_run/papo_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_run.sh"
