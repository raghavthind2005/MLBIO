#!/bin/bash
# Tiny DETACHED metrics run — submit with `sbatch smoke_metrics.sbatch`.
# Purpose: get a CLEAN wandb summary (reward/format, accuracy, clip_ratio,
# kl_prcp_loss) that a per-sample small batch reads faithfully. Detached so it
# finishes + run.finish() writes wandb-summary.json (interactive srun kept dying
# on disconnect before finalize). Tiny batch => ~20 min.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_metrics
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_smoke/metrics_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/smoke_4b_thinking.sh" \
     data.rollout_batch_size=16 worker.actor.global_batch_size=16
