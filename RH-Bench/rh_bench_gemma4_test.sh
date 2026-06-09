#!/bin/bash
# RH-Bench Gemma-4-31B TEST JOB — 5 samples per subset (10 total)
# Purpose: verify thinking mode, max_tokens, response quality, and judge quality
#          before launching the full 1000-sample benchmark.
#
# What this checks:
#   1. Gemma-4 inference works with vision
#   2. Thinking mode is ON (reasoning_content populated)
#   3. max_tokens=16384 is sufficient (no truncation)
#   4. Judge (Qwen3-32B) produces sensible verdicts
#   5. Per-sample timing → projected full-run time
#
# After job completes, review the log carefully before running the full job.

#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gemma4-test
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/rh-bench/gemma4_test_%j.log

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
SCRATCH=/iopsstor/scratch/cscs/raghavthind
CODE=$SCRATCH/code/rh-bench
mkdir -p $SCRATCH/rh-bench/results

# ─── PHASE 1: VLM INFERENCE (Gemma-4-31B, 5+5 samples) ──────────────────────
echo "========================================"
echo "PHASE 1: Gemma-4-31B Inference (limit=5)"
echo "========================================"

srun --environment=$HOME/toml/sglang_gemma4.toml bash -c "
set -e
pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'

echo '--- Starting Gemma-4-31B server ---'
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --reasoning-parser gemma4 &
VLM_PID=\$!

# Wait for server (up to 5 min)
for i in \$(seq 1 60); do
  curl -sf http://localhost:30000/health > /dev/null && echo 'VLM ready.' && break
  sleep 5
done

echo '--- Running inference (5 halu + 5 reason) ---'
python $CODE/run_inference_gemma4.py --limit 5

kill \$VLM_PID 2>/dev/null
echo '--- VLM server stopped ---'
"

# ─── PHASE 2: JUDGE (Qwen3-32B, same 10 samples) ────────────────────────────
echo ""
echo "========================================"
echo "PHASE 2: Judge Evaluation (Qwen3-32B)"
echo "========================================"

# Check which Qwen3-32B path exists
JUDGE_MODEL=""
for p in \
  "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B" \
  "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B-Instruct"; do
  if [ -d "$p" ]; then JUDGE_MODEL="$p"; break; fi
done

if [ -z "$JUDGE_MODEL" ]; then
  echo "ERROR: Qwen3-32B not found on infra. Check available models:"
  ls /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/ 2>/dev/null || echo "(path not accessible)"
  exit 1
fi
echo "Using judge: $JUDGE_MODEL"

srun --environment=$HOME/toml/sglang.toml bash -c "
echo '--- Starting Qwen3-32B judge on port 30001 ---'
python -m sglang.launch_server \
  --model-path $JUDGE_MODEL \
  --port 30001 --host 0.0.0.0 --tp 2 \
  --mem-fraction-static 0.7 &
JUDGE_PID=\$!

for i in \$(seq 1 60); do
  curl -sf http://localhost:30001/health > /dev/null && echo 'Judge ready.' && break
  sleep 5
done

echo '--- Running judge (10 samples) ---'
python $CODE/run_judge_gemma4.py --limit 10

kill \$JUDGE_PID 2>/dev/null
echo '--- Judge server stopped ---'
"

# ─── PHASE 3: SCORES ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "PHASE 3: Score Computation"
echo "========================================"

python3 - << 'PYEOF'
import json, os

f = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses_gemma4.json"
if not os.path.exists(f):
    print("ERROR: judged file not found"); exit(1)

results = json.load(open(f))
for r in results:
    if r.get("question_type") == "free_from":
        r["question_type"] = "free_form"

from collections import defaultdict
stats = defaultdict(lambda: {"correct": 0, "total": 0})
for rec in results:
    key = f"{rec['subset']}_{rec['question_type']}"
    stats[key]["total"] += 1
    if rec.get("is_correct", False):
        stats[key]["correct"] += 1

print("\n=== TEST RUN SCORES (10 samples — not statistically meaningful) ===")
for task, label in [("reason", "Reasoning"), ("halu", "Perception")]:
    print(f"\n{label}:")
    for qtype in ["multi_choice", "free_form"]:
        k = f"{task}_{qtype}"
        if stats[k]["total"] > 0:
            acc = 100 * stats[k]["correct"] / stats[k]["total"]
            print(f"  {qtype}: {acc:.0f}% ({stats[k]['correct']}/{stats[k]['total']})")

think_count = sum(1 for r in results if r.get("has_thinking"))
avg_words = sum(r.get("thinking_words", 0) for r in results) / max(1, think_count)
avg_time = sum(r.get("elapsed_s", 0) for r in results) / max(1, len(results))
print(f"\n=== QUALITY CHECKS ===")
print(f"Thinking ON:       {think_count}/{len(results)} ({100*think_count/len(results):.0f}%) — should be >80%")
print(f"Avg thinking words: {avg_words:.0f}")
print(f"Avg inference time: {avg_time:.1f}s/sample")
print(f"Projected full run: {avg_time*1000/3600:.1f}h for 1000 samples")
print(f"\nIf thinking < 80% or inference time seems wrong, DO NOT run full job yet.")
PYEOF

echo ""
echo "========================================"
echo "TEST COMPLETE — review log before full run"
echo "Log: /iopsstor/scratch/cscs/raghavthind/rh-bench/gemma4_test_${SLURM_JOB_ID}.log"
echo "========================================"
