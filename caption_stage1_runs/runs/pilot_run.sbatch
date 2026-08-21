#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-pilot
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/pilot_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Pilot 0 REAL RUN (D20: n=200 items, G=5). Gated by the smoke (D27) -- do not
# submit this until pilot_smoke.sbatch has passed at these same settings.
#
# Produces the GENERATION artifacts for measurements (a), (b) and (c):
#   pass 1  captions, G=5                      -> (b) caption length distribution
#   pass 2  blind answers from each caption    -> (c) blind-from-caption accuracy
#   pass 2b blind answers, M=3 on the subset   -> (d) variance decomposition
#   pass 3  image-conditioned answers, n=5     -> (a) THE ONE THAT SETS D32
#
# (d) additionally needs D-hat, which is Job B (GPU forwards + D34 chunking) and
# is scored later from these same artifacts -- nothing here needs re-running.
#
# On the answer budget: 16,384 is deliberately NON-BINDING. Measurement (a)
# exists to observe the true answer-length distribution, and a censored sample
# cannot report its own p99, so capping tightly here would destroy the very
# number we need to set the training cap from. The length-hacking concern that
# argues AGAINST a large budget applies to TRAINING (where D-hat is the reward,
# summed over answer tokens); it does not apply to a measurement, which has no
# reward. Requires max_model_len 32,768, since an image may take 4,096 tokens.
#
# Partition is `normal` (12h) not `debug` (1.5h): ~2,500 generations with a long
# tail at 16k will not reliably finish in the debug window.

set -euo pipefail

source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

export OUT=$CS1/pilot0
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo

echo "=== unit tests (must pass before any GPU work) ==="
cd "$CS1/code"
python3 -m unittest test_virl_pool test_cs1_prompts test_pilot_score 2>&1 | tail -4
echo

echo "=== pool gate ==="
python3 "$CS1/code/verify_pool.py" --pool "$CS1_POOL"
echo

echo "=== generation: 200 items, G=5 ==="
python3 "$CS1/code/pilot_generate.py" \
    --pool "$CS1_POOL" \
    --out "$OUT" \
    --model "$CS1_MODEL" \
    --snapshot "$CS1_SNAPSHOT" \
    --g-captions 5 \
    --m-answers 1 \
    --m-answers-subset 3 \
    --n-image-answers 5 \
    --preset untruncated \
    --max-model-len 32768 \
    --caption-max-tokens 3072 \
    --answer-max-tokens 16384 \
    --seed 0

echo
echo "=== artifacts ==="
ls -la "$OUT"
for f in captions answers_blind answers_image; do
    [ -f "$OUT/$f.jsonl" ] && echo "  $f.jsonl: $(wc -l < "$OUT/$f.jsonl") rows"
done

echo
echo "=== measurement (a): image-conditioned accuracy + D32 histogram ==="
python3 "$CS1/code/pilot_score.py" \
    --pool "$CS1_POOL" \
    --answers "$OUT/answers_image.jsonl" \
    --out "$OUT/measurement_a.json" \
    --label a

echo
echo "=== measurement (c): blind-from-caption accuracy ==="
python3 "$CS1/code/pilot_score.py" \
    --pool "$CS1_POOL" \
    --answers "$OUT/answers_blind.jsonl" \
    --out "$OUT/measurement_c.json" \
    --label c

echo
echo "=== (a) - (c): the caption's information loss ==="
python3 - <<'PY'
import json, os
out = os.environ["OUT"]
a = json.load(open(f"{out}/measurement_a.json"))
c = json.load(open(f"{out}/measurement_c.json"))
for arm in ("strict", "fallback_sensitivity"):
    ia, ic = a[arm]["overall"]["accuracy"], c[arm]["overall"]["accuracy"]
    print(f"  {arm:22s} image={ia:.3f}  blind={ic:.3f}  loss={ia - ic:+.3f}")
print()
print("  answer-length tail (the input to the training cap decision):")
meta = json.load(open(f"{out}/_meta_generate.json"))
print(f"    caption truncation rate: {meta.get('caption_truncation_rate')}")
print(f"    sampling preset        : {meta.get('sampling_preset')}")
PY

echo
echo "PILOT 0 COMPLETE"
