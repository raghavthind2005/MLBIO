#!/bin/bash
#SBATCH --job-name=ca21_t0e
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0e_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0e_%j.out

# T0e -- three open items in one container, no generation.
#   1. the torch-gated unit tests, which SKIP on the laptop and so are unverified
#   2. the upstream fit() sha256, pinned by the interpreter that will actually hash it
#   3. the B>1 row-independence gate left open by 4.17
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

# 1. Unit tests FIRST -- they need no GPU, so a failure here should not cost a model load.
#    -v so a silent skip cannot masquerade as a pass, which is the whole reason for this.
echo "=== unit tests (in-container, torch present) ==="
srun --environment="$CA21_TOML" \
    python3 -m unittest discover -s "$CA21/code" -p "test_*.py" -v 2>&1 | tail -40

echo
echo "=== fit() hash + B>1 row-independence ==="
srun --environment="$CA21_TOML" \
    python3 "$CA21/code/t0e_batch_independence.py" \
        --model "$CA21_MODEL" \
        --parquet "$CA21/data/ca21_dev.parquet" \
        --out "$CA21_POOL/t0e_batch_independence.json" \
        --n-rows "${CA21_T0E_ROWS:-4}"

echo "[job] done at $(date -Is)"
