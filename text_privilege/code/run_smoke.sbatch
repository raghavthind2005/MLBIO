#!/bin/bash
# Probe A smoke.  Submit:  sbatch $SCRATCH/text_privilege/code/run_smoke.sbatch
# Loads three 4B models sequentially (never co-resident — PAPO's vLLM sleep/wake OOM lesson).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tp_smoke
#SBATCH --output=/iopsstor/scratch/cscs/%u/text_privilege/smoke_%j.log

srun --environment="$HOME/toml/vllm011.toml" \
     bash /iopsstor/scratch/cscs/$USER/text_privilege/code/run_smoke.sh
