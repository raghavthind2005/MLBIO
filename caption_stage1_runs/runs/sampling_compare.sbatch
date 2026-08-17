#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-sampcmp
#SBATCH --partition=debug
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/sampcmp_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Diagnostic (not the frozen pipeline): does off-card sampling explain the
# answer-format non-compliance found in the Pilot-0 smoke?
set -euo pipefail
CS1=/iopsstor/scratch/cscs/raghavthind/caption_stage1
MODEL=/iopsstor/scratch/cscs/raghavthind/models/Qwen3-VL-2B-Instruct
SNAP=/iopsstor/scratch/cscs/raghavthind/hf_cache/hub/datasets--PAPOGalaxy--PAPO_ViRL39K_train/snapshots/ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb/data
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/hf_cache
mkdir -p "$CS1/logs" "$CS1/sampling_compare"

python3 "$CS1/code/sampling_compare.py" \
    --pool "$CS1/pool" --out "$CS1/sampling_compare" \
    --model "$MODEL" --snapshot "$SNAP" \
    --n-items 10 --g-captions 2 \
    --caption-max-tokens 2048 --answer-max-tokens 512
