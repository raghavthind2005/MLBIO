#!/bin/bash
#SBATCH --job-name=ca21_est
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/est_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/est_%j.out

# Unit tests for the S11 estimator. Needs torch, which exists only inside the container,
# so these cannot run on a laptop or a login node.
#
# CPU-only tensor maths on tiny tensors -- the GPU allocation is incidental to getting a
# node that can run the container at all.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

echo "=== S11 estimator + packed-layout extraction (torch-only tests) ==="
srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 -m unittest discover -s "$CA21/code" -p "test_ca21_*.py" -v

echo
echo "=== the rest of the suite, in the container this time ==="
srun --environment="$HOME/toml/verl_easyr1.toml" bash -c \
    "cd $CA21/code && python3 -m unittest \
        test_ca21_prompts test_pool_io test_build_pool \
        test_container_gate test_group_analysis test_build_verl_data 2>&1 | tail -5"

echo "[job] done at $(date -Is)"
