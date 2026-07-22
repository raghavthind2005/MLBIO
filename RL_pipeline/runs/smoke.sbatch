#!/bin/bash
# Detached PAPO smoke — submit with `sbatch smoke.sbatch`.
# Unlike interactive `srun | tee`, this survives SSH disconnects (runs in the
# background, writes to the --output log). Reads all metrics via wandb offline.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_smoke/sbatch_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/smoke_4b_thinking.sh"
