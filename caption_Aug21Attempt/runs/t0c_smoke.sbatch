#!/bin/bash
#SBATCH --job-name=ca21_t0c
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0c_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/t0c_%j.out

# T0c -- the first full training smoke. 2 steps, Arm B.
#
# NOT a result. What it proves is that rollout -> reward -> advantage -> caption term ->
# composed batch -> update_policy survives real shapes. Every component has been tested
# alone; none of them have ever run together.
#
# 4 GPUs on one node: world_size=4, and g_c=8 is a multiple of it, so the composed
# answer+caption batch divides evenly on every step regardless of how many prompts survive
# the correctness gate (main_ca21 checks this before allocating a GPU).
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "[job] config: $CA21/configs/ca21_smoke.yaml"

# G-VITATTN FIRST, on one GPU, in seconds.
#
# This gate already existed and already tested the right property when job 3177693 died on
# precisely the failure it guards -- nothing invoked it. That cost a four-GPU allocation
# through a full model load to discover that vLLM's ViT would take the flash-attn path and
# reject Qwen2.5-VL's head_dim=80. Cheapest checks run first, always.
#
# Needs CUDA (the branch under test is current_platform.is_cuda()), so it runs inside the
# allocation but on a single task. `set -e` aborts before anything expensive.
echo "=== gate: G-VITATTN (vLLM ViT attention backend) ==="
srun --environment="$CA21_TOML" --ntasks=1 --gres=gpu:1 \
    python3 "$CA21/code/container_gate.py" \
        "$(awk '{print $1}' "$CA21/patches/vllm_0_11_2/PATCHED.sha256")"

# PRE-FLIGHT. The config runs with filter_overlong_prompts=false because verl's filter
# wedges here (see the config comment and preflight_prompt_lengths.py). That is only safe
# while nothing is overlong, so prove it -- with verl's own predicate, single-process, no
# pool -- before a GPU is allocated. `set -e` aborts the job if this fails.
echo "=== pre-flight: prompt lengths ==="
srun --environment="$CA21_TOML" \
    python3 "$CA21/code/preflight_prompt_lengths.py" \
        --model "$CA21_MODEL" \
        --parquet "$CA21/data/ca21_trial_smoke.parquet" "$CA21/data/ca21_dev.parquet" \
        --format-prompt "$CA21_EASYR1/examples/format_prompt/math.jinja" \
        --max-prompt-length 2048 \
        --out "$CA21_POOL/preflight_prompt_lengths.json"

srun --environment="$CA21_TOML" \
    python3 "$CA21/code/main_ca21.py" \
        config="$CA21/configs/ca21_smoke.yaml" \
        trainer.experiment_name="t0c_smoke_${SLURM_JOB_ID}"

echo "[job] done at $(date -Is)"
