#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-pool
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/pool_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Pool construction for Pilot 0 (Step 1). CPU-only -- no GPU, no model.
# Runs inside the container because the login node has neither pyarrow nor
# mathruler (it is a bare Python 3.6).
#
# Usage:
#   sbatch build_pool.sbatch --report-only        # gradeable fraction only
#   sbatch build_pool.sbatch                      # + draw the seeded pool

set -euo pipefail

CS1=/iopsstor/scratch/cscs/raghavthind/caption_stage1
CODE=$CS1/code
OUT=$CS1/pool
REPO=/iopsstor/scratch/cscs/raghavthind/MLBIO

mkdir -p "$CS1/logs" "$OUT"

# Code provenance: $CODE is a plain cp target, not a git repo, so the SHA has to
# come from the clone the files were copied FROM. Without this the manifest
# silently records "unknown".
export CS1_GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "code git sha: $CS1_GIT_SHA"

echo "=== environment ==="
python3 -c "import sys; print('python', sys.version.split()[0])"
python3 -c "import pyarrow; print('pyarrow', pyarrow.__version__)"
python3 -c "import mathruler; print('mathruler OK')"
echo

echo "=== unit tests (must pass before the parser touches real data) ==="
cd "$CODE"
python3 -m unittest test_virl_pool -v 2>&1 | tail -5
echo

echo "=== pool construction ==="
python3 "$CODE/build_pool_cluster.py" --out "$OUT" "$@"

echo
echo "=== artifacts ==="
ls -la "$OUT"
