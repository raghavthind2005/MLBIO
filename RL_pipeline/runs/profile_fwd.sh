#!/bin/bash
# Forward-pass profiler (runs INSIDE the container via profile_fwd.sbatch).
#
# PURE DIAGNOSIS — changes nothing: py-spy only reads stack traces of the running
# python workers; the training runs ONE step and saves nothing (save_freq=-1).
# Goal: find the hot frames behind the ~64-80 s/it compute_log_probs forward.
#
# It backgrounds a py-spy sampler on the busiest python worker while running one
# small step (offload-off / enforce_eager / filter-off all inherited from the
# smoke script), then aggregates.
set -x
SCRATCH=/iopsstor/scratch/cscs/$USER
OUT=$SCRATCH/runs/papo_smoke/pyspy_${SLURM_JOB_ID:-manual}.txt
: > "$OUT"

# py-spy is in the image; install if somehow missing (compute node has internet)
command -v py-spy >/dev/null 2>&1 || pip install --quiet py-spy 2>/dev/null || true

sample_pyspy() {
  for i in $(seq 1 150); do
    # busiest python proc = the worker currently launching the forward kernels
    PID=$(ps -eo pid,%cpu,comm --sort=-%cpu | awk '$3 ~ /python/ {print $1; exit}')
    if [ -n "$PID" ]; then
      { echo "=== sample $i pid=$PID cpu-busiest $(date +%T) ==="; py-spy dump --pid "$PID" 2>&1; } >> "$OUT"
    fi
    sleep 12
  done
}
sample_pyspy &
SAMPLER=$!

# one small, representative step (experience micro-batch = 16 seqs = same forward
# shape as the full run); filter off + eager so startup is fast
bash "$SCRATCH/code/runs/smoke_4b_thinking.sh" \
    data.rollout_batch_size=16 worker.actor.global_batch_size=16 \
    data.filter_overlong_prompts=false trainer.max_steps=1

kill "$SAMPLER" 2>/dev/null
echo "=================================================="
echo "PY-SPY SAMPLES WRITTEN TO: $OUT"
echo "Aggregate hot frames with:"
echo "  grep -A2 'Thread .*active' $OUT | grep -vE 'Thread|--' | sort | uniq -c | sort -rn | head -30"
