#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-verify
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/verify_%j.log

# ─── Config — EDIT these paths to match where you put babyVision on the cluster ──
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"        # contains verify_capture.py
DATA_DIR="$REPO/repo/data/babyvision_data"                       # meta_data.jsonl + images/
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
MODEL_PATH='$MODEL_PATH'
REPO='$REPO'
DATA_DIR='$DATA_DIR'

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
  echo 'ERROR: VLM server never became healthy after 15 minutes. Aborting.'
  kill \$VLM_PID 2>/dev/null
  exit 1
fi

python \$REPO/verify_capture.py --port 30000 --data-dir \$DATA_DIR --max-tokens 4096

kill \$VLM_PID 2>/dev/null
echo 'Verify done.'
"
