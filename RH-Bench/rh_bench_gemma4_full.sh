#!/bin/bash
# RH-Bench Gemma-4-31B FULL JOB — 1000 samples, 2 nodes (VLM + judge concurrent)
#
# Node 1: Gemma-4-31B-it inference (tp=4, all 4 GPUs)
# Node 2: Qwen3-32B judge (tp=2)
#
# Run this ONLY after rh_bench_gemma4_test.sh passes all quality checks.
# Monitor progress:
#   tail -f /iopsstor/scratch/cscs/raghavthind/rh-bench/gemma4_full_<JOBID>.log
#   watch -n 30 'python3 /iopsstor/scratch/cscs/raghavthind/code/rh-bench/progress_gemma4.py'

#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=11:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=rh-bench-gemma4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/rh-bench/gemma4_full_%j.log

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface
SCRATCH=/iopsstor/scratch/cscs/raghavthind
CODE=$SCRATCH/code/rh-bench
mkdir -p $SCRATCH/rh-bench/results

# Get node assignments
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
VLM_NODE="${NODES[0]}"
JUDGE_NODE="${NODES[1]}"
echo "VLM node:   $VLM_NODE"
echo "Judge node: $JUDGE_NODE"

# Detect judge model path
JUDGE_MODEL=""
for p in \
  "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B" \
  "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B-Instruct"; do
  if [ -d "$p" ]; then JUDGE_MODEL="$p"; break; fi
done
[ -z "$JUDGE_MODEL" ] && echo "ERROR: Qwen3-32B not found" && exit 1
echo "Judge model: $JUDGE_MODEL"

# ─── NODE 1: Gemma-4-31B Inference ───────────────────────────────────────────
srun --nodes=1 --ntasks=1 -w "$VLM_NODE" \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "

pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'

python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --reasoning-parser gemma4 \
  --disable-overlap-schedule \
  --enable-metrics &
VLM_PID=\$!

echo 'Waiting for VLM server...'
for i in \$(seq 1 60); do
  curl -sf http://localhost:30000/health > /dev/null && echo 'VLM ready.' && break
  sleep 5
done

python $CODE/run_inference_gemma4.py  # full 1000 samples (no --limit)

kill \$VLM_PID 2>/dev/null
echo 'VLM done.'
" &
VLM_TASK=$!

# ─── NODE 2: Qwen3-32B Judge ─────────────────────────────────────────────────
srun --nodes=1 --ntasks=1 -w "$JUDGE_NODE" \
  --environment=$HOME/toml/sglang.toml bash -c "

# Wait for inference to produce output (check every 2 min)
RESULTS=/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses_gemma4.json
echo 'Waiting for inference results to appear...'
until [ -f \"\$RESULTS\" ]; do sleep 120; done
echo 'Results file found, starting judge server...'

python -m sglang.launch_server \
  --model-path $JUDGE_MODEL \
  --port 30001 --host 0.0.0.0 --tp 2 \
  --mem-fraction-static 0.7 &
JUDGE_PID=\$!

for i in \$(seq 1 60); do
  curl -sf http://localhost:30001/health > /dev/null && echo 'Judge ready.' && break
  sleep 5
done

# Wait until inference is fully done (all 1000 entries written)
echo 'Waiting for all 1000 inference results...'
while true; do
  N=\$(python3 -c \"import json; d=json.load(open('\$RESULTS')); print(len(d))\" 2>/dev/null || echo 0)
  echo \"  Inference progress: \$N/1000\"
  [ \"\$N\" -ge 1000 ] && break
  sleep 120
done
echo 'Inference complete. Running judge...'

python $CODE/run_judge_gemma4.py  # full run (no --limit)

kill \$JUDGE_PID 2>/dev/null
echo 'Judge done.'
" &
JUDGE_TASK=$!

# Wait for both nodes
wait $VLM_TASK $JUDGE_TASK

# ─── PHASE 3: SCORES ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "FINAL SCORES"
echo "========================================"
python3 $CODE/compute_scores.py

echo "Full results: $SCRATCH/rh-bench/results/judged_responses_gemma4.json"
