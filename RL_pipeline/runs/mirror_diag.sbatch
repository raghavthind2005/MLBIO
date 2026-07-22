#!/bin/bash
# Full-mirror DIAGNOSTIC — submit with `sbatch mirror_diag.sbatch`.
# Runs the canonical papo_run.sh (identical to the real run) but only 2 steps and
# no checkpoints. Detached (survives disconnect). Read metrics from the wandb
# offline binary via diagnostics/read_wandb_metrics.py.
#
# Signals to sign off on: response_length/clip_ratio (truncation at 8192 with
# model-card sampling), mean response length, rollout wall-time, OOM/no-OOM, and
# reward/format/accuracy/kl_prcp.
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_mirror
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_run/mirror_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_run.sh" \
     trainer.max_steps=2 trainer.save_freq=-1
