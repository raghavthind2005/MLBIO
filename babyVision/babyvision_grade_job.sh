#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-grade
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/grade_%j.log

# ─── Once-and-for-all grading (grade.py: judge on the model's final answer) ──────
# Launches the Qwen3 judge server, then grades EVERY condition dir under BASE in one
# pass (grade.py loops conditions internally). Format-agnostic; writes
# results_run{N}_graded.jsonl with `grade` + `grade_raw`. Resumable.
# ─────────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL_PATH="/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-32B"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
BASE="${BASE:-/iopsstor/scratch/cscs/raghavthind/babyvision}"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
# ─────────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang.toml bash -c "
JUDGE_MODEL_PATH='$JUDGE_MODEL_PATH'
REPO='$REPO'
BASE='$BASE'

# Stock container ships a matching torch + sgl_kernel; do NOT reinstall torch.
python -m sglang.launch_server \
  --model-path \$JUDGE_MODEL_PATH \
  --port 30001 --host 0.0.0.0 --tp 4 \
  --mem-fraction-static 0.8 \
  --watchdog-timeout 600 \
  --enable-metrics &
JUDGE_PID=\$!

echo 'Waiting for judge server (up to 15 min)...'
SERVER_READY=0
for i in \$(seq 1 180); do
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

python \$REPO/grade.py --port 30001 --base \$BASE

EXIT_CODE=\$?
kill \$JUDGE_PID 2>/dev/null
echo \"Grading done (exit code \$EXIT_CODE).\"
exit \$EXIT_CODE
"
