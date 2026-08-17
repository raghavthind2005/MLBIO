#!/bin/bash
# PAPO 2B-Thinking 60-step real run (normal, 12h). Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_run.sbatch
# ~20 min/step -> one 12h slot does ~33 steps + base val; RESUBMIT the same line to
# auto-resume the remainder + final val (~2 slots total for 60 steps).
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_2b_8k_run
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_8k_run/run_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_8k_run.sh"
