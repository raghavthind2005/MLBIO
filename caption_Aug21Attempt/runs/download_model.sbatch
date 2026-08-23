#!/bin/bash
#SBATCH --job-name=ca21_model
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/model_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/model_%j.out

# Download Qwen2.5-VL-3B-Instruct (O1) at a pinned revision and verify it.
#
# TARGET IS THE SHARED /capstor STORE, NOT SCRATCH. Scratch lost this project's
# data once (2026-08-11, all 21 PAPO checkpoints), and unlike the dataset a model
# is what every later measurement is defined against. It is also where the
# existing models live, including the Qwen2.5-VL-7B-Instruct downloaded for the
# DeepEyes baseline -- same family, same convention.
#
# Root is hardcoded: Slurm copies this script to /var/spool/slurmd/ before
# running it, so $(dirname "$0") is the spool directory (job 3163721 died there).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/fetch_model.py" \
        --repo "$CA21_MODEL_REPO" \
        --revision "$CA21_MODEL_REV" \
        --target "$CA21_MODEL" \
        --out "$CA21_POOL/model_provenance.json" \
        --expect-gb 8.0

echo "[job] done at $(date -Is)"
