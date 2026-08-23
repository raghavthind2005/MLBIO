# Single sourced environment for every caption_Aug21Attempt run script.
#
#   source "$(dirname "$0")/_env.sh"
#
# WHY THIS FILE EXISTS: per-script environment blocks drift. In the previous
# attempt one runner silently omitted VLLM_WORKER_MULTIPROC_METHOD=spawn and
# died with "Cannot re-initialize CUDA in forked subprocess" only after loading
# the model. One sourced file cannot drift from itself.
#
# Override anything by exporting it before sourcing.

# --- paths ---------------------------------------------------------------
export CA21=${CA21:-/iopsstor/scratch/cscs/raghavthind/caption_aug21}
export CA21_REPO=${CA21_REPO:-/iopsstor/scratch/cscs/raghavthind/MLBIO_ca21}

# Dataset: Vision-SR1-47K, PINNED revision. Never track `main` -- a silent
# upstream re-upload would change the pool underneath every measurement.
export CA21_DATASET=${CA21_DATASET:-LMMs-Lab-Turtle/Vision-SR1-47K}
export CA21_DATASET_REV=${CA21_DATASET_REV:-2900b038f4aaa72f6b92795c1ee3ab29b7d509b6}

# HF cache lives on scratch. This is a deliberate exception to the "durable
# store" rule: scratch lost this project's data once (2026-08-11), but the
# dataset is re-downloadable in minutes from the pinned revision above.
# Checkpoints and run outputs are NOT reproducible and must never live here.
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/raghavthind/hf_cache}

export CA21_POOL=${CA21_POOL:-$CA21/pool}

# --- runtime -------------------------------------------------------------
# REQUIRED for vLLM in this container; without it the engine core dies with
# "Cannot re-initialize CUDA in forked subprocess".
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# NEVER set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- it crashes the
# vLLM CuMemAllocator on this stack (learned the hard way in the PAPO line).

export PYTHONUNBUFFERED=1

# --- provenance ----------------------------------------------------------
# $CA21/code is a plain cp target, not a git repo, so the SHA must come from
# the clone the files were copied from. Without this, artifacts record "unknown".
export CA21_GIT_SHA="$(git -C "$CA21_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

mkdir -p "$CA21/logs" "$CA21_POOL"

echo "[_env] git     = $CA21_GIT_SHA"
echo "[_env] dataset = $CA21_DATASET @ $CA21_DATASET_REV"
echo "[_env] hf_home = $HF_HOME"
echo "[_env] pool    = $CA21_POOL"
