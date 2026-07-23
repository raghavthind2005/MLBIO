#!/bin/bash
# PAPO 2B-Thinking 8K memory re-check (debug). Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_smoke.sbatch
# PASS = reaches step-2 wake, NO OOM, peak max_memory_reserved < ~92/95. Also yields
# T_step@8192 for the real-run estimate.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_2b_8k
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_8k_smoke/smoke_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_8k_smoke.sh"
