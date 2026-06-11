#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=hallusion-tool-gemma4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/hallusionbench/logs/tool_%j.log

mkdir -p /iopsstor/scratch/cscs/raghavthind/hallusionbench/results_tool
mkdir -p /iopsstor/scratch/cscs/raghavthind/hallusionbench/logs

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c '

pip install -q "kernels==0.3.0"
pip install -q "transformers>=5.10.1"
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"

python -m sglang.launch_server \
  --model-path /capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --watchdog-timeout 600 \
  --reasoning-parser gemma4 \
  --disable-overlap-schedule \
  --enable-metrics &
VLM_PID=$!

echo "Waiting for VLM server..."
for i in $(seq 1 60); do
  curl -sf http://localhost:30000/health > /dev/null && echo "VLM ready." && break
  sleep 5
done

python /iopsstor/scratch/cscs/raghavthind/code/hallusionbench_repo/image_toolCalling/run_eval_tool.py \
  --port 30000 \
  --fraction 0.30 \
  --out /iopsstor/scratch/cscs/raghavthind/hallusionbench/results_tool/tool_results.jsonl

kill $VLM_PID 2>/dev/null
echo "Done."
'
