#!/bin/bash
# =============================================================================
# run_accuracy_curve.sh — parse all 3 condition logs → curves.csv + H verdict
#
# Runs on login node (pure Python, no GPU/container needed).
# But also works inside the container if needed.
#
# Usage (on login node — no sbatch needed):
#   bash runs/analysis/run_accuracy_curve.sh
#
# Or as sbatch (inside container for consistency):
#   sbatch runs/analysis/run_accuracy_curve.sh
# =============================================================================
# (Uncomment SBATCH lines to run as a job)
##SBATCH --account=a0174
##SBATCH --partition=normal
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=0
##SBATCH --cpus-per-task=4
##SBATCH --time=00:10:00
##SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
##SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out

set -euo pipefail

LOG_DIR=/iopsstor/scratch/cscs/raghavthind/runs/logs
ANALYSIS=/iopsstor/scratch/cscs/raghavthind/runs/analysis
OUT=${ANALYSIS}/curves.csv
mkdir -p "$ANALYSIS"

export PYTHONUNBUFFERED=1

# ── Find the most recent log for each condition ──────────────────────────────
LOG_FULL=$(ls -t   "$LOG_DIR"/stage1_full-*.out    2>/dev/null | head -1 || true)
LOG_LLM=$(ls -t    "$LOG_DIR"/stage1_llm_only-*.out 2>/dev/null | head -1 || true)
LOG_VIT=$(ls -t    "$LOG_DIR"/stage1_vit_only-*.out 2>/dev/null | head -1 || true)

# Also check stage1_full in its own directory (old location used by stage1_full.sh)
if [[ -z "$LOG_FULL" ]]; then
    LOG_FULL=$(ls -t /iopsstor/scratch/cscs/raghavthind/runs/stage1_full/slurm-*.out \
               2>/dev/null | head -1 || true)
fi

echo "Logs found:"
echo "  full     : ${LOG_FULL:-NOT FOUND}"
echo "  llm_only : ${LOG_LLM:-NOT FOUND}"
echo "  vit_only : ${LOG_VIT:-NOT FOUND}"

# ── Step 0: dump raw metric block from step 6 of the full-condition log ──────
# (run this first to verify key names match _KEY_ALIASES in accuracy_curve.py)
if [[ -n "$LOG_FULL" ]]; then
    echo ""
    echo "=== RAW METRIC BLOCK (step 6, full condition) ==="
    python3 "$ANALYSIS/accuracy_curve.py" \
        --dump-step 6 \
        --log "full=$LOG_FULL"
    echo "=== END RAW BLOCK ==="
fi

# ── Parse all available logs ──────────────────────────────────────────────────
LOG_ARGS=()
[[ -n "$LOG_FULL" ]] && LOG_ARGS+=(--log "full=$LOG_FULL")
[[ -n "$LOG_LLM"  ]] && LOG_ARGS+=(--log "llm_only=$LOG_LLM")
[[ -n "$LOG_VIT"  ]] && LOG_ARGS+=(--log "vit_only=$LOG_VIT")

if [[ ${#LOG_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no log files found in $LOG_DIR"
    exit 1
fi

python3 "$ANALYSIS/accuracy_curve.py" "${LOG_ARGS[@]}" --out "$OUT"
echo ""
echo "CSV written to: $OUT"
