#!/bin/bash
#SBATCH --job-name=ca21_pool
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/pool_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/pool_%j.out

# Build the dev / trial / eval pools from the VERIFIED snapshot.
#
# CPU-only: the images are never decoded, only the light columns are read, and
# the sole heavy dependency is the container's `mathruler` for the gradeability
# rule. Minutes, not hours.
#
# Root is hardcoded because Slurm copies this script to /var/spool/slurmd/ before
# running it, so $(dirname "$0") points at the spool directory (job 3163721 died
# on exactly that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/build_pool.py" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21_POOL" \
        --seed "${CA21_SEED:-0}" \
        --n-trial "${CA21_N_TRIAL:-22000}" \
        --n-eval-final "${CA21_N_EVAL_FINAL:-8000}" \
        --n-eval-monitor "${CA21_N_EVAL_MONITOR:-1000}" \
        --n-dev "${CA21_N_DEV:-300}" \
        --target "${CA21_TARGET:-eligible}"

echo "[job] done at $(date -Is)"
