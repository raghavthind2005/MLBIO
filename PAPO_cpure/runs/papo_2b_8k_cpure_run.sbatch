#!/bin/bash
# PAPO_CPURE (faithful paper PAPO_G-2B) 60-step real run (normal, 12h). Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_cpure_run.sbatch
# ~20 min/step (no recompute) -> ~2 slots. Queue the resume unattended with:
#   sbatch --dependency=afterany:<jobid> $SCRATCH/code/runs/papo_2b_8k_cpure_run.sbatch
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cpure_run
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_2b_8k_cpure_run/run_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_2b_8k_cpure_run.sh"
