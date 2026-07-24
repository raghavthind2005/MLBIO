#!/bin/bash
# PAPO_FIXED (Eq.2) memory+timing smoke on debug. Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_papofix_smoke.sbatch
# Tests the RECOMPUTE=True path (extra masked grad-forward): no OOM at util 0.45, T_step, and
# that kl_prcp is now MAXIMIZED (proof PAPO trains). ~40-60 min (RECOMPUTE ~1.5-2x slower/step).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papofix_smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_8k_papofix_smoke/smoke_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_8k_papofix_smoke.sh"
