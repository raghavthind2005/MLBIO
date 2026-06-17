#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-infer-gemma4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/infer_%j.log

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
DATA_DIR="$REPO/repo/data/babyvision_data"
OUT_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/results_standard"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
# How many passes to run in this job (1 = one pass per job if wall-time is tight)
N_PASSES="${N_PASSES:-3}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR" "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
MODEL_PATH='$MODEL_PATH'
REPO='$REPO'
DATA_DIR='$DATA_DIR'
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

echo 'Waiting for VLM server (up to 15 min for CUDA graph capture)...'
SERVER_READY=0
for i in \$(seq 1 180); do
  if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
    echo \"VLM ready after \$((i*5))s.\"
    SERVER_READY=1
    break
  fi
  sleep 5
done

if [ \$SERVER_READY -eq 0 ]; then
  echo 'ERROR: VLM server never became healthy. Aborting.'
  kill \$VLM_PID 2>/dev/null
  exit 1
fi

python \$REPO/run_infer.py \
  --port 30000 \
  --data-dir \$DATA_DIR \
  --out-dir  \$OUT_DIR \
  --n-passes $N_PASSES

EXIT_CODE=\$?
kill \$VLM_PID 2>/dev/null
echo \"Inference done (exit code \$EXIT_CODE).\"
exit \$EXIT_CODE
"
