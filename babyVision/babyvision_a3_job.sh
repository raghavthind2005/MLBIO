#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-a3-forcedlong
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/a3_%j.log

# ─── A3 · forced-LONG (s1 single-trace budget forcing) ──────────────────────────
# Spike-gated: launch server → validate mechanism on 10 samples → only run the
# full 388 if the spike PASSES (exit 0). Single pass. Image shown ONCE (no reinject).
# ────────────────────────────────────────────────────────────────────────────────
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
DATA_DIR="$REPO/repo/data/babyvision_data"
OUT_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/results_a3_forced_long"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
SPIKE_N="${SPIKE_N:-10}"
# Set SKIP_SPIKE=1 to bypass the gate (only after a prior PASS).
SKIP_SPIKE="${SKIP_SPIKE:-0}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR" "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
MODEL_PATH='$MODEL_PATH'
REPO='$REPO'
DATA_DIR='$DATA_DIR'
OUT_DIR='$OUT_DIR'
SPIKE_N='$SPIKE_N'
SKIP_SPIKE='$SKIP_SPIKE'

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

# ── Spike gate ────────────────────────────────────────────────────────────────
if [ \"\$SKIP_SPIKE\" != \"1\" ]; then
  echo '=== SPIKE: validating single-trace budget forcing ==='
  python \$REPO/run_infer_a3.py \
    --port 30000 --model-path \$MODEL_PATH \
    --data-dir \$DATA_DIR --spike \$SPIKE_N
  SPIKE_RC=\$?
  echo \"Spike exit code: \$SPIKE_RC\"
  if [ \$SPIKE_RC -ne 0 ]; then
    echo 'SPIKE FAILED — not launching full run. Inspect the log above.'
    kill \$VLM_PID 2>/dev/null
    exit \$SPIKE_RC
  fi
  echo '=== SPIKE PASSED — launching full run ==='
fi

# ── Full run ────────────────────────────────────────────────────────────────────
python \$REPO/run_infer_a3.py \
  --port 30000 --model-path \$MODEL_PATH \
  --data-dir \$DATA_DIR --out-dir \$OUT_DIR

EXIT_CODE=\$?
kill \$VLM_PID 2>/dev/null
echo \"A3 done (exit code \$EXIT_CODE).\"
exit \$EXIT_CODE
"
