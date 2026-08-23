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

# Backbone: Qwen2.5-VL-3B-Instruct (O1), PINNED. This is the small backbone of
# this research direction -- Vision-SR1's primary 3B, PAPO's paper primary
# family -- and critically it is a NON-thinking Instruct model. Note Qwen2.5-VL
# has no Thinking variant at all; "-Instruct" is the only release. The runaway
# chains that truncated 82-95% at PAPO's own 2048 budget were a Qwen3-VL trait
# we imported, not a property of this literature.
export CA21_MODEL_REPO=${CA21_MODEL_REPO:-Qwen/Qwen2.5-VL-3B-Instruct}
export CA21_MODEL_REV=${CA21_MODEL_REV:-66285546d2b821cf421d4f5eb2576359d3770cd3}
# The DURABLE store, matching where Qwen2.5-VL-7B-Instruct already lives.
export CA21_MODEL=${CA21_MODEL:-/capstor/store/cscs/swissai/a0174/models/Qwen2.5-VL-3B-Instruct}

# HF cache. NOTE: the cluster environment already exports HF_HOME, so in practice
# this default does not apply -- job 3163760 resolved it to
# /iopsstor/scratch/cscs/raghavthind/huggingface, not the hf_cache named here.
# Recorded rather than forced, since re-pointing it would re-download 4.35 GB for
# no gain; every artifact records the real path it used.
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/raghavthind/hf_cache}

export CA21_POOL=${CA21_POOL:-$CA21/pool}

# Container. NOT ~/toml/verl_easyr1.toml: ours adds a bind-mount that overlays a patched
# vllm/attention/layer.py, and that must not appear underneath the PAPO or DeepEyes lines,
# which share the other file. See patches/vllm_0_11_2/README.md.
export CA21_TOML=${CA21_TOML:-$CA21/runs/ca21_vllm.toml}
export CA21_PATCH_LAYER=${CA21_PATCH_LAYER:-$CA21/patches/vllm_0_11_2/layer.py}

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
echo "[_env] toml    = $CA21_TOML"

# The mount source must exist and be the file we patched. If it is missing, the container
# engine may start anyway with the stock layer.py and the run dies 90 s later in the ViT --
# so this is checked here, not discovered there.
if [ ! -f "$CA21_PATCH_LAYER" ]; then
    echo "[_env] FATAL: patched layer.py absent at $CA21_PATCH_LAYER" >&2
    exit 1
fi
_want=$(awk '{print $1}' "$CA21/patches/vllm_0_11_2/PATCHED.sha256")
_got=$(sha256sum "$CA21_PATCH_LAYER" | awk '{print $1}')
if [ "$_want" != "$_got" ]; then
    echo "[_env] FATAL: patched layer.py hash mismatch" >&2
    echo "[_env]   expected $_want" >&2
    echo "[_env]   found    $_got" >&2
    exit 1
fi
echo "[_env] patch   = layer.py OK ($(echo "$_got" | cut -c1-16)...)"
