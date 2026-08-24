#!/bin/bash
#SBATCH --job-name=ca21_t0a
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0a_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0a_%j.out

# T0a -- first execution of the caption-distortion stack against a real model.
#
# NOT the training smoke: no Ray, no FSDP, no vLLM, no fit(). One GPU is enough
# because this is four forward passes per item, not a training step.
#
# `normal` and not `debug`: debug caps at 1:30 but is meant for short turnaround,
# and this needs a model load plus generation. 40 min is generous for 4 items.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$CA21_TOML" \
    python3 "$CA21/code/t0a_component_smoke.py" \
        --model "$CA21_MODEL" \
        --parquet "$CA21/data/ca21_dev.parquet" \
        --out "$CA21_POOL/t0a_component_smoke.json" \
        --n-items "${CA21_T0A_ITEMS:-4}" \
        --max-caption-tokens "${CA21_T0A_CAPTOK:-256}" \
        --max-answer-tokens "${CA21_T0A_ANSTOK:-256}"

echo "[job] done at $(date -Is)"
