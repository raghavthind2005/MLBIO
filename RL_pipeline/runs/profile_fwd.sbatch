#!/bin/bash
# Detached forward-pass profiler — submit with `sbatch profile_fwd.sbatch`.
# PURE DIAGNOSIS: py-spy reads stacks only; the 1-step run saves nothing.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_profile
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_smoke/profile_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/profile_fwd.sh"
