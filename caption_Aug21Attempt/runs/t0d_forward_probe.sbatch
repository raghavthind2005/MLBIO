#!/bin/bash
#SBATCH --job-name=ca21_t0d
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0d_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0d_%j.out

# T0d -- which forward is wrong, packed or padded? See code/t0d_forward_probe.py.
#
# NO GENERATION, which is what makes this cheap: T0b spent 628 s generating captions and
# trajectories before it could score anything. This needs neither. Six items x 4 forwards,
# plus a ~60 s model load.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$CA21_TOML" \
    python3 "$CA21/code/t0d_forward_probe.py" \
        --model "$CA21_MODEL" \
        --parquet "$CA21/data/ca21_dev.parquet" \
        --out "$CA21_POOL/t0d_forward_probe.json" \
        --n-items "${CA21_T0D_ITEMS:-6}"

echo "[job] done at $(date -Is)"
