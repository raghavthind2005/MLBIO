# Shared environment for every caption_stage1 run script.
#
#   source "$(dirname "$0")/_env.sh"
#
# WHY THIS EXISTS: sampling_compare.sbatch was written separately from
# pilot_smoke.sbatch and silently omitted VLLM_WORKER_MULTIPROC_METHOD=spawn,
# so it died with "Cannot re-initialize CUDA in forked subprocess" after
# loading the model. Per-script environment blocks drift; one sourced file
# cannot. Modelled on VLM-CapCurriculum's training/_env.sh.
#
# Override anything by exporting it before sourcing.

# --- paths ---------------------------------------------------------------
export CS1=${CS1:-/iopsstor/scratch/cscs/raghavthind/caption_stage1}
export CS1_REPO=${CS1_REPO:-/iopsstor/scratch/cscs/raghavthind/MLBIO}
# Backbone: Qwen3-VL-4B-Instruct (D33, supersedes D4's 2B). Deliberately the
# /capstor/store copy, not a scratch copy: scratch has already lost data on this
# project once (Aug 11), and store is the durable filesystem.
export CS1_MODEL=${CS1_MODEL:-/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct}
export CS1_SNAPSHOT=${CS1_SNAPSHOT:-/iopsstor/scratch/cscs/raghavthind/hf_cache/hub/datasets--PAPOGalaxy--PAPO_ViRL39K_train/snapshots/ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb/data}
export CS1_POOL=${CS1_POOL:-$CS1/pool}
export CS1_PRESERVE_ROOT=${CS1_PRESERVE_ROOT:-/capstor/store/cscs/swissai/a0174/caption_stage1_ckpts}

# --- runtime -------------------------------------------------------------
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/raghavthind/hf_cache}

# REQUIRED for vLLM in this container: without it the engine core dies with
# "Cannot re-initialize CUDA in forked subprocess".
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# NEVER set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here -- it crashes
# the vLLM CuMemAllocator on this stack (learned in the PAPO line).

export PYTHONUNBUFFERED=1

# --- provenance ----------------------------------------------------------
# $CS1/code is a plain cp target, not a git repo, so the SHA must come from the
# clone the files were copied from. Without this, artifacts record "unknown".
export CS1_GIT_SHA="$(git -C "$CS1_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

mkdir -p "$CS1/logs"

echo "[_env] git=$CS1_GIT_SHA"
echo "[_env] model=$CS1_MODEL"
echo "[_env] pool=$CS1_POOL"
