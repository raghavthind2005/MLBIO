#!/bin/bash
# PAPO_CPURE memory+correctness smoke on debug. Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_cpure_smoke.sbatch
# Validates util 0.55 / 8192 / RECOMPUTE=False / entropy OFF: no OOM, kl_prcp MAXIMIZED. ~30-45 min.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cpure_smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_8k_cpure_smoke/smoke_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_8k_cpure_smoke.sh"
