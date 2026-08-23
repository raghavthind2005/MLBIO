#!/bin/bash
#SBATCH --job-name=ca21_data
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/data_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/data_%j.out

# Materialise the pool manifest as parquet files verl loads directly.
# CPU only -- no GPU work, no vLLM, so the shared toml is fine.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/build_verl_data.py" \
        --pool "$CA21_POOL" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21/data"

echo "[job] done at $(date -Is)"
