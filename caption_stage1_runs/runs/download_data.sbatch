#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-data
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/data_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Re-download the ViRL39K training set (decision D5).
#
# WHY THIS EXISTS: the copy verified on 2026-07-21 was purged from scratch. The
# HF cache kept the snapshot directory and its six symlinks, but the blob store
# behind them is empty -- so the paths still LOOK present to `ls` while every
# read fails. Any future check must resolve the symlinks (`test -e`), not just
# list the directory.
#
# Pinned to the same revision the earlier work used, so the pool is built on
# byte-identical data.

set -euo pipefail

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/hf_cache
REV=ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb

mkdir -p /iopsstor/scratch/cscs/raghavthind/caption_stage1/logs

python3 - <<PY
from huggingface_hub import snapshot_download

for repo, rev in [
    ("PAPOGalaxy/PAPO_ViRL39K_train", "$REV"),
    ("PAPOGalaxy/PAPO_MMK12_test", None),
]:
    print(f"--- {repo} (revision={rev or 'main'}) ---", flush=True)
    path = snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, max_workers=8)
    print("  ->", path, flush=True)
PY

echo
echo "=== verify the symlinks RESOLVE (the check that was missing before) ==="
D=/iopsstor/scratch/cscs/raghavthind/hf_cache/hub/datasets--PAPOGalaxy--PAPO_ViRL39K_train
fail=0
for f in "$D"/snapshots/*/data/*.parquet; do
  if [ -e "$f" ]; then
    echo "OK     $(basename "$f")  $(stat -Lc%s "$f") bytes"
  else
    echo "BROKEN $(basename "$f")"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "FAILED: dangling symlinks remain"; exit 1; }

echo
echo "=== total size ==="
du -sh "$D"
