#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-judge-qwen3
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/judge_%j.log

# ─── Config ───────────────────────────────────────────────────────────────────
JUDGE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
# Override per condition, e.g.:
#   sbatch --export=RESULTS_DIR=.../results_a0_nothink,PASSES=1 babyvision_judge_job.sh
#   sbatch --export=RESULTS_DIR=.../results_a3_forced_long,PASSES=1 babyvision_judge_job.sh
RESULTS_DIR="${RESULTS_DIR:-/iopsstor/scratch/cscs/raghavthind/babyvision/results_standard}"
PASSES="${PASSES:-1 2 3}"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang.toml bash -c "
JUDGE_MODEL_PATH='$JUDGE_MODEL_PATH'
REPO='$REPO'
RESULTS_DIR='$RESULTS_DIR'
PASSES='$PASSES'

pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'

python -m sglang.launch_server \
  --model-path \$JUDGE_MODEL_PATH \
  --port 30001 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --watchdog-timeout 300 \
  --enable-metrics &
JUDGE_PID=\$!

echo 'Waiting for judge server...'
SERVER_READY=0
for i in \$(seq 1 60); do
  if curl -sf http://localhost:30001/health > /dev/null 2>&1; then
    echo \"Judge ready after \$((i*5))s.\"
    SERVER_READY=1
    break
  fi
  sleep 5
done

if [ \$SERVER_READY -eq 0 ]; then
  echo 'ERROR: Judge server never became healthy. Aborting.'
  kill \$JUDGE_PID 2>/dev/null
  exit 1
fi

python \$REPO/run_judge.py \
  --port 30001 \
  --results-dir \$RESULTS_DIR \
  --passes \$PASSES

EXIT_CODE=\$?
kill \$JUDGE_PID 2>/dev/null
echo \"Judging done (exit code \$EXIT_CODE).\"
exit \$EXIT_CODE
"
