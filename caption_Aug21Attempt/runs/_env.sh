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

# --- EasyR1 / verl -------------------------------------------------------
# Vision-SR1 is built on EasyR1 (its config schema is EasyR1's exactly), so this line
# needs EasyR1's verl -- NOT the upstream-verl layout that PAPO and DeepEyes vendor, which
# has different module paths entirely.
#
# PINNED to the commit every line number in code/ was verified against. This matters more
# than it looks: the local RL_SeeingToThinking checkout carries uncommitted edits
# (freeze_language_model, Qwen3-VL patches) that shift fsdp_workers.py by 14 lines. Reading
# citations off THAT tree and running against a different one is how "verified" claims
# quietly stop being true. Kept as its own clone so no other project's edits can reach it.
#
# NOTE /iopsstor/.../code/EasyR1 is an EMPTY SHELL -- __pycache__ dirs, zero .py files.
# Do not point at it; job 3172930 died on exactly that.
export CA21_EASYR1=${CA21_EASYR1:-/iopsstor/scratch/cscs/raghavthind/EasyR1_ca21}
export CA21_EASYR1_REV=${CA21_EASYR1_REV:-dd71bbd}
export PYTHONPATH="$CA21_EASYR1${PYTHONPATH:+:$PYTHONPATH}"

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

# --- durable logging -----------------------------------------------------
# Scratch on this cluster has been wiped without warning. The PAPO runs survived only
# because their numbers were also in wandb -- but PAPO ran WANDB_MODE=offline
# (PAPO_fixed/runs/papo_2b_8k_papofix_run.sh:25), which writes to the SAME filesystem that
# gets wiped and only leaves the cluster on a later manual `wandb sync`. A wipe before that
# sync loses everything.
#
# [V] Clariden COMPUTE nodes reach api.wandb.ai directly (job on nid006895, HTTP 404 from
# the root path = the connection succeeded). So we log ONLINE and the data is off-cluster
# within seconds of being produced.
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-caption_aug21}
# Pinned rather than left to the account default, so a later default change cannot scatter
# runs across entities and make them hard to find. [V] resolved from the API, 2026-08-24.
export WANDB_ENTITY=${WANDB_ENTITY:-rthind-university-of-maryland}
export WANDB_DIR=${WANDB_DIR:-$CA21/wandb}
# Keep the local mirror too: two copies in different failure domains, not one.
export CA21_RECORDS=${CA21_RECORDS:-$CA21/records}
mkdir -p "$WANDB_DIR" "$CA21_RECORDS"

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

# G-EASYR1. Checked here, before a GPU is allocated: a wrong or dirty verl means every
# line number cited in code/ describes different code than the code that runs.
if [ ! -f "$CA21_EASYR1/verl/workers/actor/dp_actor.py" ]; then
    echo "[_env] FATAL: no EasyR1 verl at $CA21_EASYR1" >&2
    echo "[_env]   git clone https://github.com/hiyouga/EasyR1.git \$CA21_EASYR1" >&2
    echo "[_env]   git -C \$CA21_EASYR1 checkout $CA21_EASYR1_REV" >&2
    exit 1
fi
_er_head=$(git -C "$CA21_EASYR1" rev-parse --short HEAD 2>/dev/null || echo unknown)
_er_dirty=$(git -C "$CA21_EASYR1" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$_er_head" != "$CA21_EASYR1_REV" ] || [ "$_er_dirty" != "0" ]; then
    echo "[_env] FATAL: EasyR1 is $_er_head with $_er_dirty modified files;" >&2
    echo "[_env]   expected $CA21_EASYR1_REV clean. Every citation in code/ was verified" >&2
    echo "[_env]   against that commit; running another one voids them." >&2
    exit 1
fi
echo "[_env] easyr1  = $_er_head (clean) -> $CA21_EASYR1"

# G-WANDB. Checked HERE, before a GPU is allocated, because the failure it prevents is the
# expensive one: a run that trains for hours and then turns out to have logged nowhere
# durable. Credentials are absent by default on this cluster (verified 2026-08-24), which
# is almost certainly why the PAPO line fell back to offline.
if [ "${CA21_REQUIRE_WANDB:-1}" = "1" ] && [ "$WANDB_MODE" = "online" ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "${NETRC:-$HOME/.netrc}"; then
        echo "[_env] FATAL: WANDB_MODE=online but no credentials found." >&2
        echo "[_env]   Fix ONE of these, then re-submit:" >&2
        echo "[_env]     wandb login                 # writes ~/.netrc, do this once" >&2
        echo "[_env]     export WANDB_API_KEY=...    # or supply it per-run" >&2
        echo "[_env]   To run WITHOUT durable logging (not for R1):" >&2
        echo "[_env]     export CA21_REQUIRE_WANDB=0" >&2
        exit 1
    fi
    echo "[_env] wandb   = online -> $WANDB_PROJECT (records mirror: $CA21_RECORDS)"
else
    echo "[_env] wandb   = ${WANDB_MODE} (durable logging NOT guaranteed)"
fi
