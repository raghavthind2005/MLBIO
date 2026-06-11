#!/bin/bash
# ─── Config — change these for different runs ─────────────────────────────────
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
FRACTION=0.30          # 0.30 for 30% subset, 1.0 for full dataset
SEED=42
MAX_TOKENS=16384
REPO="/iopsstor/scratch/cscs/raghavthind/code/hallusionbench_repo/image_toolCalling"
OUT_DIR="/iopsstor/scratch/cscs/raghavthind/hallusionbench/results_normal"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/hallusionbench/logs"
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=hallusion-normal-gemma4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/hallusionbench/logs/normal_%j.log

mkdir -p "$OUT_DIR" "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
MODEL_PATH='$MODEL_PATH'
MAX_TOKENS='$MAX_TOKENS'
FRACTION='$FRACTION'
SEED='$SEED'
REPO='$REPO'
OUT_DIR='$OUT_DIR'

pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'

python -m sglang.launch_server \
  --model-path \$MODEL_PATH \
  --port 30000 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --watchdog-timeout 600 \
  --reasoning-parser gemma4 \
  --disable-overlap-schedule \
  --enable-metrics &
VLM_PID=\$!

echo 'Waiting for VLM server...'
for i in \$(seq 1 60); do
  curl -sf http://localhost:30000/health > /dev/null && echo 'VLM ready.' && break
  sleep 5
done

python \$REPO/run_eval.py \
  --port 30000 \
  --fraction \$FRACTION \
  --seed \$SEED \
  --max-tokens \$MAX_TOKENS \
  --out \$OUT_DIR/raw_results.jsonl

kill \$VLM_PID 2>/dev/null
echo 'Done.'
"
