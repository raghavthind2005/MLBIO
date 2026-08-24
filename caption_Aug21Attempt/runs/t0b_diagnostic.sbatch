#!/bin/bash
#SBATCH --job-name=ca21_t0b
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:20:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0b_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0b_%j.out

# T0b -- Phase 0 (is the forward correct?) + Phase 1 (does D rank captions correctly?).
#
# Budget, from T0a's MEASURED rates rather than a guess: caption gen 11.9 s each,
# trajectory gen 9.3 s each, scoring ~0.27 s per (caption, trajectory) pair.
#   generation: 16 items x (3 captions x 11.9 + 2 trajectories x 9.3) ~= 15 min
#   scoring   : 16 x 2 trajectories x 9 variants x 0.27 s          ~=  1.5 min
#   model load                                                      ~=  1 min
# ~18 min against a 1:20 cap, so a slow node or longer captions still fit.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$CA21_TOML" \
    python3 "$CA21/code/t0b_diagnostic.py" \
        --model "$CA21_MODEL" \
        --parquet "$CA21/data/ca21_dev.parquet" \
        --out "$CA21_POOL/t0b_diagnostic.json" \
        --n-items "${CA21_T0B_ITEMS:-16}" \
        --g-c "${CA21_T0B_GC:-3}" \
        --m "${CA21_T0B_M:-2}"

echo "[job] done at $(date -Is)"
