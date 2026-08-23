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
# EXPECT_POOL_SHA below exactly -- if it does not, something non-deterministic has
# crept in and that is worth knowing before any of it is trained on.
#
# That claim used to live only in this comment. It is now an ASSERTION (below),
# because the rebuild OVERWRITES pool_manifest.json -- the same file the parquets
# in $CA21/data were sliced from. A silent divergence would orphan every parquet
# while this job printed a clean compliance report. 4.6: the failures in this
# project all produced correct-looking output right up to the point of failure.
#
# Root hardcoded: Slurm copies this script to /var/spool/slurmd/ before running
# it, so $(dirname "$0") is the spool directory (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

# Pinned to the manifest the parquets in $CA21/data were actually sliced from
# (data/verl_data_provenance.json:pool_manifest_sha256, job 3169110).
EXPECT_POOL_SHA=9a109667b1065bed6440dd0489ef83ce9a76d3774c29870b9a2f9d963f055465
POOL_LOG="$CA21/logs/fmt_pool_${SLURM_JOB_ID}.log"

echo "=== rebuilding pool (manifest now carries problem text) ==="
srun --environment="$CA21_TOML" \
    python3 "$CA21/code/build_pool.py" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21_POOL" \
        --seed "${CA21_SEED:-0}" \
        --n-trial "${CA21_N_TRIAL:-22000}" \
        --n-eval-final "${CA21_N_EVAL_FINAL:-8000}" \
        --n-eval-monitor "${CA21_N_EVAL_MONITOR:-1000}" \
        --n-dev "${CA21_N_DEV:-300}" \
        --n-trial-smoke "${CA21_N_TRIAL_SMOKE:-2000}" \
        --target "${CA21_TARGET:-eligible}" 2>&1 | tee "$POOL_LOG"

# Sizes are passed explicitly above rather than left to build_pool.py's defaults,
# so a later default change cannot silently rebuild a different pool under the
# same script and still be compared against the pin.
GOT_POOL_SHA=$(sed -n 's/^\[done\] manifest sha256 \([0-9a-f]\{64\}\)$/\1/p' "$POOL_LOG" | tail -1)
if [ "$GOT_POOL_SHA" != "$EXPECT_POOL_SHA" ]; then
    echo "[FATAL] pool rebuild is NOT reproducible, or the sizes changed."
    echo "  expected: $EXPECT_POOL_SHA"
    echo "  got:      ${GOT_POOL_SHA:-<no hash line found in $POOL_LOG>}"
    echo "  pool_manifest.json has just been OVERWRITTEN with a pool the parquets"
    echo "  in $CA21/data were NOT built from. Rebuild the data before using either."
    exit 1
fi
echo "[gate] pool rebuild reproduced $EXPECT_POOL_SHA -- parquets in \$CA21/data still match"

echo
echo "=== sighted prompt compliance check ==="
srun --environment="$CA21_TOML" \
    python3 "$CA21/code/format_check.py" \
        --pool "$CA21_POOL" \
        --model "$CA21_MODEL" \
        --provenance "$CA21_POOL/dataset_provenance.json" \
        --out "$CA21_POOL/format_check.json" \
        --split dev \
        --limit "${CA21_FMT_LIMIT:-0}" \
        --n-draws "${CA21_FMT_DRAWS:-8}" \
        --max-tokens "${CA21_FMT_MAXTOK:-8192}"

echo "[job] done at $(date -Is)"
