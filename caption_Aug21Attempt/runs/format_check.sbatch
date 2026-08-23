#!/bin/bash
#SBATCH --job-name=ca21_fmt
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/fmt_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/fmt_%j.out

# S7 gate: does Qwen2.5-VL-3B-Instruct comply with the sighted prompt?
#
# Rebuilds the pool FIRST, because the manifest now carries the problem text and
# the compliance check needs it. The rebuild is deterministic at the same seed
# and the manifest hash covers only the row selection, so this must reproduce
# 63164939e6ca0ef58026fac8bc690e7fc217dabb06ad52570a1e510acfcbfe57 exactly --
# if it does not, something non-deterministic has crept in and that is worth
# knowing before any of it is trained on.
#
# Root hardcoded: Slurm copies this script to /var/spool/slurmd/ before running
# it, so $(dirname "$0") is the spool directory (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

echo "=== rebuilding pool (manifest now carries problem text) ==="
srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/build_pool.py" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21_POOL" \
        --seed "${CA21_SEED:-0}" \
        --n-trial "${CA21_N_TRIAL:-5000}" \
        --n-eval "${CA21_N_EVAL:-1000}" \
        --n-dev "${CA21_N_DEV:-300}" \
        --target "${CA21_TARGET:-eligible}"

echo
echo "=== sighted prompt compliance check ==="
srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/format_check.py" \
        --pool "$CA21_POOL" \
        --model "$CA21_MODEL" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21_POOL/format_check.json" \
        --split dev \
        --limit "${CA21_FMT_LIMIT:-50}" \
        --n-draws "${CA21_FMT_DRAWS:-1}" \
        --max-tokens "${CA21_FMT_MAXTOK:-8192}"

echo "[job] done at $(date -Is)"
