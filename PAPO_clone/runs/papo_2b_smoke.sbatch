#!/bin/bash
# PAPO 2B-Thinking smoke launcher (debug partition). Submit with:
#   sbatch $SCRATCH/code/runs/papo_2b_smoke.sbatch
# Runs base val -> step1 -> step2 -> final val at PAPO's exact 2B recipe (util 0.6, full
# offload, 2048 resp). PASS = reaches step-2 wake with NO OOM. Also yields T_val + T_step
# for the real-run length estimate ( ~= 2*T_val + 204*T_step ).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_2b_smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_smoke/smoke_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_smoke.sh"
