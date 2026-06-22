#!/bin/bash
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=babyvision-b2cot
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/babyvision/logs/b2cot_%j.log

# ─── B2' · paired no-reinject + chain-of-thought (reuses B1''s turn 1) ─────────
# The paired control for B1': identical turn 1 (loaded from B1''s run via
# --turn1-from), reasoning folded into the turn-2 USER message, but the image is
# NOT re-shown. So B1' vs B2' differ ONLY in whether turn 2 sees the image — a
# clean paired contrast with zero turn-1 sampling variance.
# PREREQUISITE: B1' (results_b1cot_reinject) must be finished first.
# The spike generates its own turn 1 (format check only); the full run reuses B1''s.
# Spike-gated; set SKIP_SPIKE=1 after a PASS, SPIKE_ONLY=1 to stop after the spike.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH="/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
REPO="/iopsstor/scratch/cscs/raghavthind/code/babyvision"
DATA_DIR="$REPO/repo/data/babyvision_data"
OUT_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/results_b2cot_noreinject"
T1_FROM="/iopsstor/scratch/cscs/raghavthind/babyvision/results_b1cot_reinject"
LOG_DIR="/iopsstor/scratch/cscs/raghavthind/babyvision/logs"
SPIKE_N="${SPIKE_N:-5}"           # verify on 5 samples first
SKIP_SPIKE="${SKIP_SPIKE:-0}"
SPIKE_ONLY="${SPIKE_ONLY:-0}"     # set 1 to stop after a passing spike (no full run)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR" "$LOG_DIR"

srun --nodes=1 --ntasks=1 \
  --environment=$HOME/toml/sglang_gemma4.toml bash -c "
MODEL_PATH='$MODEL_PATH'
REPO='$REPO'
DATA_DIR='$DATA_DIR'
OUT_DIR='$OUT_DIR'
T1_FROM='$T1_FROM'
SPIKE_N='$SPIKE_N'
SKIP_SPIKE='$SKIP_SPIKE'
SPIKE_ONLY='$SPIKE_ONLY'

pip install -q 'kernels==0.3.0'
pip install -q 'transformers>=5.10.1'
pip install -q --index-url https://download.pytorch.org/whl/cu130/ \
    'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0'

# ── Retry loop ──────────────────────────────────────────────────────────────────
# A triton sliding-window-attention CUDA illegal-memory-access can kill the server
# mid-run on long sequences under batching. One IMA tears down the whole CUDA
# context, so every later request fails. On each attempt we relaunch a FRESH server
# (new CUDA context) and re-run; run_infer_b.py resumes by skipping the taskIds
# already completed without error, so each attempt only does what's left. We stop
# once all 388 are done (or after MAX_ATTEMPTS).
MAX_ATTEMPTS=6
DID_SPIKE=0
DONE=0
for attempt in \$(seq 1 \$MAX_ATTEMPTS); do
  echo \"=== Attempt \$attempt/\$MAX_ATTEMPTS ===\"

  python -m sglang.launch_server \
    --model-path \$MODEL_PATH \
    --port 30000 --host 0.0.0.0 --tp 4 \
    --mem-fraction-static 0.8 \
    --watchdog-timeout 1800 \
    --reasoning-parser gemma4 \
    --disable-overlap-schedule \
    --enable-metrics &
  VLM_PID=\$!

  echo 'Waiting for VLM server (up to 15 min)...'
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
    echo 'WARNING: VLM server never became healthy this attempt; retrying.'
    kill \$VLM_PID 2>/dev/null; sleep 10; continue
  fi

  # Spike gate runs ONCE. It generates its own turn 1 (format/no-image check only);
  # the full run below reuses B1''s turn 1 via --turn1-from.
  if [ \"\$SKIP_SPIKE\" != \"1\" ] && [ \$DID_SPIKE -eq 0 ]; then
    echo '=== SPIKE: validating 2-turn no-reinject+CoT protocol ==='
    python \$REPO/run_infer_b.py \
      --port 30000 --model-path \$MODEL_PATH \
      --data-dir \$DATA_DIR \
      --spike \$SPIKE_N --no-reinject --fold-reasoning
    SPIKE_RC=\$?
    echo \"Spike exit code: \$SPIKE_RC\"
    if [ \$SPIKE_RC -ne 0 ]; then
      echo 'SPIKE FAILED — inspect log above.'
      kill \$VLM_PID 2>/dev/null; exit \$SPIKE_RC
    fi
    DID_SPIKE=1
    echo '=== SPIKE PASSED ==='
    if [ \"\$SPIKE_ONLY\" = \"1\" ]; then
      echo 'SPIKE_ONLY=1 — stopping after spike (inspect log, then resubmit with SKIP_SPIKE=1).'
      kill \$VLM_PID 2>/dev/null
      exit 0
    fi
    echo '=== launching full run (reusing B1 turn 1) ==='
  fi

  # Full run — reuse B1''s turn 1 (identical), generate only turn 2 (no image).
  # Resumes; only does the taskIds still missing/errored.
  python \$REPO/run_infer_b.py \
    --port 30000 --model-path \$MODEL_PATH \
    --data-dir \$DATA_DIR --out-dir \$OUT_DIR \
    --no-reinject --fold-reasoning --turn1-from \$T1_FROM
  kill \$VLM_PID 2>/dev/null; sleep 10

  DONE=\$(python -c \"import json; recs=[json.loads(l) for l in open('\$OUT_DIR/results_run1.jsonl') if l.strip()]; print(len({r['taskId'] for r in recs if 'error' not in r}))\" 2>/dev/null || echo 0)
  echo \"Completed \$DONE/388 after attempt \$attempt.\"
  if [ \"\$DONE\" -ge 388 ]; then
    echo 'All 388 completed.'; break
  fi
  echo 'Not all done (server likely crashed mid-run) — relaunching fresh and resuming.'
done

if [ \"\$DONE\" -ge 388 ]; then
  echo \"B2' done: 388/388.\"
  exit 0
else
  echo \"B2' INCOMPLETE: only \$DONE/388 after \$MAX_ATTEMPTS attempts.\"
  exit 1
fi
"
