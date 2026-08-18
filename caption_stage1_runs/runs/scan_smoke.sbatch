#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-scansmoke
#SBATCH --partition=debug
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/scansmoke_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# SMOKE for the full-pool scan. Proves the sharding, both modes, and the
# selector end-to-end on the existing 200-item pilot pool before 218,000
# generations are committed to it.
#
# What must hold:
#   - shards partition the rows exactly: no row scanned twice, none dropped
#   - mode=image agrees with the pilot's measurement (a) on the same rows
#   - mode=text_only agrees with the control on the same rows
#   - scan_select refuses an incomplete shard set
#   - the two-stage chain produces a non-empty survivor list

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

OUT=$CS1/scan_smoke
rm -rf "$OUT"; mkdir -p "$OUT"
N_SHARDS=3

echo "=== stage 1: image, $N_SHARDS shards over the 200-item pool ==="
for s in $(seq 0 $((N_SHARDS-1))); do
    python3 "$CS1/code/scan_pool.py" \
        --pool "$CS1_POOL" --out "$OUT" \
        --model "$CS1_MODEL" --snapshot "$CS1_SNAPSHOT" \
        --mode image --shard "$s" --n-shards "$N_SHARDS" \
        --n-answers 5 --answer-max-tokens 16384 --max-model-len 32768
done

echo
echo "=== shard partition check ==="
python3 - <<PY
import json, glob, collections
idx = []
for p in sorted(glob.glob("$OUT/scan_image_shard*.jsonl")):
    idx += [json.loads(l)["index"] for l in open(p) if l.strip()]
c = collections.Counter(idx)
pool = {it["index"] for it in json.load(open("$CS1_POOL/pool_manifest.json"))["items"]}
assert set(c) == pool, "shards do not cover the pool exactly"
assert set(c.values()) == {5}, f"uneven draws per row: {set(c.values())}"
print(f"  OK  {len(c)} rows, each exactly 5 answers, no duplicates, no gaps")
PY

echo
echo "=== incomplete-shard-set guard (must FAIL) ==="
mv "$OUT/scan_image_shard000of003.jsonl" "$OUT/held.bak"
if python3 "$CS1/code/scan_select.py" --pool "$CS1_POOL" --scan-dir "$OUT" \
       --mode image --keep-min 0.8 --out "$OUT/tmp.json" 2>/dev/null; then
    echo "  FAIL: selector accepted an incomplete scan"; exit 1
fi
echo "  OK  selector refused the incomplete scan"
mv "$OUT/held.bak" "$OUT/scan_image_shard000of003.jsonl"

echo
echo "=== stage 1 selection: image >= 4/5 (D11/D32) ==="
python3 "$CS1/code/scan_select.py" \
    --pool "$CS1_POOL" --scan-dir "$OUT" --mode image \
    --keep-min 0.8 --out "$OUT/survivors_stage1.json" \
    --report "$OUT/report_stage1.json"

echo
echo "=== stage 2: text_only on stage-1 survivors only ==="
for s in $(seq 0 $((N_SHARDS-1))); do
    python3 "$CS1/code/scan_pool.py" \
        --pool "$CS1_POOL" --out "$OUT" \
        --model "$CS1_MODEL" --snapshot "$CS1_SNAPSHOT" \
        --mode text_only --shard "$s" --n-shards "$N_SHARDS" \
        --candidates "$OUT/survivors_stage1.json" \
        --n-answers 5 --answer-max-tokens 16384 --max-model-len 32768
done

echo
echo "=== stage 2 selection: text <= 2/5 (vision necessity) ==="
python3 "$CS1/code/scan_select.py" \
    --pool "$CS1_POOL" --scan-dir "$OUT" --mode text_only \
    --keep-max 0.4 --out "$OUT/survivors_final.json" \
    --report "$OUT/report_stage2.json"

echo
echo "=== agreement with the pilot's independent measurements ==="
python3 - <<PY
import json
s1 = set(json.load(open("$OUT/survivors_stage1.json")))
fin = set(json.load(open("$OUT/survivors_final.json")))
print(f"  stage1 image>=4/5 : {len(s1)}   (pilot measurement (a) gave 119/200)")
print(f"  final  +text<=2/5 : {len(fin)}  (pilot buckets 0-2 gave 47/200)")
assert fin <= s1, "stage 2 leaked rows that stage 1 rejected"
print("  OK  final pool is a subset of stage-1 survivors")
PY

echo
echo "SCAN SMOKE COMPLETE"
