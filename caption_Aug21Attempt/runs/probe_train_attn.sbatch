#!/bin/bash
#SBATCH --job-name=ca21_h3
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/h3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/raghavthind/caption_aug21/logs/h3_%j.out

# H3: does verl's HF training path load and step Qwen2.5-VL with a usable ViT kernel?
#
# Three arms: verl's current global flash_attention_2 (expected to fail on head_dim=80),
# the per-subconfig dict fix, and global sdpa as a control. Full rationale in
# code/probe_train_attn.py.
#
# Uses the SHARED toml, not ca21_vllm.toml, on purpose: the S10 patch only touches
# vLLM's rollout path, and this probe must observe the HF path as verl will actually
# find it -- unmodified.
#
# Root hardcoded: Slurm copies this to /var/spool/slurmd/ (job 3163721 died on that).

set -euo pipefail

CA21_ROOT=/iopsstor/scratch/cscs/raghavthind/caption_aug21
source "$CA21_ROOT/runs/_env.sh"

echo "[job] $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun --environment="$HOME/toml/verl_easyr1.toml" \
    python3 "$CA21/code/probe_train_attn.py" \
        --model "$CA21_MODEL" \
        --out "$CA21/pool/h3_train_attn.json"

echo "[job] done at $(date -Is)"
