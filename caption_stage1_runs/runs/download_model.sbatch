#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-dl
#SBATCH --partition=debug
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/download_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Download Qwen3-VL-2B-Instruct (decision D4), ~4.3 GB.
#
# Must run on a COMPUTE node inside the container: the login node has no
# huggingface_hub and no outbound path for this, while the debug partition does
# have internet.

set -euo pipefail

DEST=/iopsstor/scratch/cscs/raghavthind/models/Qwen3-VL-2B-Instruct
export HF_HOME=/iopsstor/scratch/cscs/raghavthind/hf_cache

mkdir -p /iopsstor/scratch/cscs/raghavthind/caption_stage1/logs "$DEST"

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

dest = "/iopsstor/scratch/cscs/raghavthind/models/Qwen3-VL-2B-Instruct"
path = snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    local_dir=dest,
    max_workers=8,
)
print("downloaded to:", path)
PY

echo
echo "=== contents ==="
ls -la "$DEST"

echo
echo "=== provenance values we depend on (D23, D24) ==="
python3 - <<'PY'
import json
base = "/iopsstor/scratch/cscs/raghavthind/models/Qwen3-VL-2B-Instruct"
for name in ("generation_config.json", "preprocessor_config.json"):
    with open(f"{base}/{name}") as fh:
        cfg = json.load(fh)
    print(f"--- {name} ---")
    print(json.dumps(cfg, indent=2)[:700])
PY
