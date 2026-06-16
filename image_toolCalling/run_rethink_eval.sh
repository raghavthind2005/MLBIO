#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=rethink-eval
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/rethink_eval_%j.log

export HF_HOME=/iopsstor/scratch/cscs/raghavthind/huggingface

CODE=/iopsstor/scratch/cscs/raghavthind/code/hallusionbench_repo/image_toolCalling

srun --environment=$HOME/toml/sglang_gemma4.toml bash -c '
pip install -q "kernels==0.3.0"
pip install -q "transformers>=5.10.1"
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"

python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --enable-metrics &

SERVER_PID=$!

echo "Waiting for sglang server..."
until curl -s http://localhost:30000/v1/models > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready."

echo "=== DRY RUN (5 samples) ==="
python '"$CODE"'/run_eval_rethink.py --port 30000 --dry-run 5

echo "=== FULL RUN ==="
python '"$CODE"'/run_eval_rethink.py \
  --port 30000 \
  --fraction 0.30 \
  --out '"$CODE"'/results_rethink/rethink_results.jsonl

kill $SERVER_PID
'
