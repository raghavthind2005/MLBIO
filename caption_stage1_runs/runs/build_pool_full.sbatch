#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-poolfull
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/poolfull_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Build the FULL candidate pool (every eligible row, not a 200-item draw) for
# the two-stage scan. CPU-only.
#
# Writes to $CS1/pool_full, NEVER $CS1/pool. The pilot pool is referenced by
# manifest_sha256 in every Pilot 0 artifact (_meta_generate.json,
# measurement_*.json); overwriting it would silently break the provenance chain
# linking those measurements to the rows they were computed on.

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

OUT=$CS1/pool_full
mkdir -p "$OUT"

echo "=== unit tests ==="
cd "$CS1/code"
python3 -m unittest test_virl_pool 2>&1 | tail -4
echo

echo "=== building full candidate pool (n-items 0 = all eligible) ==="
python3 "$CS1/code/build_pool_cluster.py" --out "$OUT" --n-items 0 --n-subset 0 --seed 0

echo
echo "=== pool gate ==="
python3 "$CS1/code/verify_pool.py" --pool "$OUT"

echo
echo "=== summary ==="
python3 - <<PY
import json
m = json.load(open("$OUT/pool_manifest.json"))
items = m["items"]
from collections import Counter
print(f"  rows           : {len(items):,}")
print(f"  manifest sha256: {m.get('manifest_sha256','?')[:16]}...")
print(f"  by answer_fmt  : {dict(Counter(i.get('answer_fmt') for i in items))}")
print(f"  by kind        : {dict(Counter(i.get('kind') for i in items))}")
pilot = json.load(open("$CS1/pool/pool_manifest.json"))
p = {i['index'] for i in pilot['items']}
f = {i['index'] for i in items}
print(f"  pilot rows also in full pool: {len(p & f)}/{len(p)}")
PY

echo
echo "FULL POOL BUILT"
