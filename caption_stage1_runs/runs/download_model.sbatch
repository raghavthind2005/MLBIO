#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-dl
#SBATCH --partition=debug
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/download_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Fetch a HuggingFace model into the models tree.
#
# NOT NEEDED FOR THE CURRENT BACKBONE. D33 moved us to Qwen3-VL-4B-Instruct,
# which was already present at
#   /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
# (verified 2026-08-17: both shards, 8.88 GB, vocab 151,936). This script is
# kept as the record of how Qwen3-VL-2B-Instruct (D4) was fetched, and as a
# working tool if another backbone is ever needed.
#
# Must run on a COMPUTE node inside the container: the login node has no
# huggingface_hub and no outbound path for this, while the debug partition does
# have internet.
#
# Usage:
#   sbatch download_model.sbatch                          # default repo below
#   sbatch download_model.sbatch Qwen/Qwen3-VL-8B-Instruct

set -euo pipefail

REPO_ID=${1:-Qwen/Qwen3-VL-2B-Instruct}
DEST=/iopsstor/scratch/cscs/raghavthind/models/$(basename "$REPO_ID")
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/hf_cache

mkdir -p /iopsstor/scratch/cscs/raghavthind/caption_stage1/logs "$DEST"

echo "repo: $REPO_ID"
echo "dest: $DEST"

REPO_ID="$REPO_ID" DEST="$DEST" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["REPO_ID"],
    local_dir=os.environ["DEST"],
    max_workers=8,
)
print("downloaded to:", path)
PY

echo
echo "=== contents ==="
ls -la "$DEST"

echo
echo "=== provenance values we depend on (D23, D24) ==="
DEST="$DEST" python3 - <<'PY'
import json, os
base = os.environ["DEST"]
for name in ("generation_config.json", "preprocessor_config.json"):
    with open(f"{base}/{name}") as fh:
        cfg = json.load(fh)
    print(f"--- {name} ---")
    print(json.dumps(cfg, indent=2)[:700])
PY
