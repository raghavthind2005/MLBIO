#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=rh-bench
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/rh-bench/bench_%j.log

# RH-Bench evaluation — arXiv:2505.21523
# VLM:   Qwen3-VL-4B-Thinking  (GPU 0,     port 30000)
# Judge: Qwen3.6-27B            (GPUs 1-3,  port 30001)

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface

srun --environment=$HOME/toml/sglang.toml bash -c '

# VLM on GPU 0 only (--base-gpu-id works inside container; CUDA_VISIBLE_DEVICES does not)
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking \
  --port 30000 --host 0.0.0.0 --tp 1 --base-gpu-id 0 --mem-fraction-static 0.7 --reasoning-parser qwen3-thinking &
VLM_PID=$!

# Judge on GPUs 1-3
python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B \
  --port 30001 --host 0.0.0.0 --tp 3 --base-gpu-id 1 --mem-fraction-static 0.8 &
JUDGE_PID=$!

echo "Waiting for both servers..."
for i in $(seq 1 120); do
  VLM_UP=$(curl -sf http://localhost:30000/health && echo 1 || echo 0)
  JUDGE_UP=$(curl -sf http://localhost:30001/health && echo 1 || echo 0)
  [ "$VLM_UP" = "1" ] && [ "$JUDGE_UP" = "1" ] && echo "Both servers ready!" && break
  sleep 5 && echo "  waiting ($i/120)..."
done

echo "=== Phase 1: VLM Inference ==="
python /iopsstor/scratch/cscs/raghavthind/code/rh-bench/run_inference.py

echo "=== Phase 2: Judge Scoring ==="
python /iopsstor/scratch/cscs/raghavthind/code/rh-bench/run_judge.py

echo "=== Phase 3: Compute Scores ==="
python /iopsstor/scratch/cscs/raghavthind/code/rh-bench/compute_scores.py

kill $VLM_PID $JUDGE_PID 2>/dev/null
echo "Done."
'
