#!/bin/bash
# Forward-pass profiler v2 (runs INSIDE the container via profile_fwd.sbatch).
#
# PURE DIAGNOSIS — changes nothing: py-spy reads stacks only; the run saves nothing.
# v1 bug: ps top-CPU sampled Ray's log-monitor + the main proc (blocked on ray.get),
# never the workers. v2 targets the GPU-using worker pids (nvidia-smi) and uses
# --nonblocking (no process freeze -> no observer slowdown). Self-aggregates at the end.
set -x
SCRATCH=/iopsstor/scratch/cscs/$USER
OUT=$SCRATCH/runs/papo_smoke/pyspy_${SLURM_JOB_ID:-manual}.txt
: > "$OUT"

command -v py-spy >/dev/null 2>&1 || pip install --quiet py-spy 2>/dev/null || true

gpu_pids() {
  # the FSDP worker processes doing the forward = the python procs holding GPU memory
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -cd '0-9\n' | grep -E '^[0-9]+$' | sort -u
  # fallback: ray actor processes
  pgrep -f "WorkerDict" 2>/dev/null
}

sample() {
  for i in $(seq 1 500); do
    for PID in $(gpu_pids | sort -u); do
      { echo "=== s$i pid=$PID $(date +%T) ==="; py-spy dump --pid "$PID" --nonblocking 2>&1; } >> "$OUT"
    done
    sleep 4
  done
}
sample &
SAMPLER=$!

# one small, representative step; filter off + eager for fast startup
bash "$SCRATCH/code/runs/smoke_4b_thinking.sh" \
    data.rollout_batch_size=16 worker.actor.global_batch_size=16 \
    data.filter_overlong_prompts=false trainer.max_steps=1

kill "$SAMPLER" 2>/dev/null; sleep 1

echo "=================================================="
echo "PY-SPY raw dumps: $OUT"
echo "=== capture quality ==="
echo "  valid active-thread samples: $(grep -ac 'Thread .*active' "$OUT")"
echo "  perm-denied:                 $(grep -ac 'Permission Denied' "$OUT")"
echo "  pids sampled:"; grep -a '=== s' "$OUT" | grep -oE 'pid=[0-9]+' | sort | uniq -c | sort -rn | head
echo "=== TOP FORWARD FRAMES (ray/idle/import noise filtered) ==="
grep -aoE "[A-Za-z0-9_<>.]+ \([^()]*\.py:[0-9]+\)" "$OUT" \
  | grep -avE "importlib|_bootstrap|frozen|<module>|threading\.py|ray/_private|tqdm|selectors|log_monitor|gc_collect|asyncio|socket\.py|queue\.py" \
  | sort | uniq -c | sort -rn | head -40
