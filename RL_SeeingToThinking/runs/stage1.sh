#!/bin/bash
# =============================================================================
# Stage 1 (Visual Perception) RLVR — 3 FREEZE CONDITIONS from ONE script.
#
# The ONLY thing that differs between conditions is which component is fine-tuned
# (the two freeze flags below). Every training-relevant setting — data, model,
# GRPO/PPO hyperparameters, reward, batch sizes, lr, KL, epochs, seed, the
# conv->matmul fix, checkpointing/logging — is byte-identical across conditions.
# This guarantees a clean ablation: same rollouts/data/init, only the learning
# target (which params receive gradients) changes.
#
#   Usage:  sbatch --job-name=stage1_<cond>  runs/stage1.sh  <cond>
#   <cond> in:
#     full      freeze_vision_tower=false freeze_language_model=false   (LLM + ViT)   [= Condition 1]
#     llm_only  freeze_vision_tower=true  freeze_language_model=false   (LLM only; ViT frozen)
#     vit_only  freeze_vision_tower=false freeze_language_model=true    (ViT only; LLM frozen)
#
# NOTE on the forward: freezing only stops gradients/updates; the frozen component
# still runs in the forward, so generation/reward/log-probs use the FULL model in
# all three conditions. The conv->matmul patch trains the ViT correctly (grads flow
# to proj.weight). freeze_language_model is our added flag (fsdp_workers.py).
# Checkpoints go to a per-condition dir so the three runs never collide.
# =============================================================================
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/logs/%x-%j.out

set -euo pipefail

# ---- the ONLY per-condition difference ----
COND="${1:?usage: sbatch --job-name=stage1_<cond> runs/stage1.sh <full|llm_only|vit_only>}"
case "$COND" in
  full)     FREEZE_VIT=false; FREEZE_LLM=false ;;
  llm_only) FREEZE_VIT=true;  FREEZE_LLM=false ;;
  vit_only) FREEZE_VIT=false; FREEZE_LLM=true  ;;
  *) echo "ERROR: condition must be full | llm_only | vit_only (got '$COND')"; exit 1 ;;
esac

EASYR1=/iopsstor/scratch/cscs/raghavthind/code/EasyR1
VLMCC=/iopsstor/scratch/cscs/raghavthind/code/VLM-CapCurriculum
DATA=/capstor/store/cscs/swissai/a0174/datasets/VLM-CapCurriculum-Perception-Data
MODEL=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
RUN=/iopsstor/scratch/cscs/raghavthind/runs/stage1_${COND}
mkdir -p "$RUN" /iopsstor/scratch/cscs/raghavthind/runs/logs

export PYTHONPATH=$EASYR1:${PYTHONPATH:-}
export WANDB_MODE=offline
export WANDB_DIR=$RUN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "===== CONDITION: $COND  (freeze_vision_tower=$FREEZE_VIT  freeze_language_model=$FREEZE_LLM) ====="
cd "$EASYR1"
python3 -m verl.trainer.main \
    config="$VLMCC/training/configs/config.yaml" \
    data.train_files="$DATA/perception_difficulty_curriculum.jsonl" \
    data.val_files=hiyouga/geometry3k@test \
    data.image_dir="$DATA/images" \
    data.format_prompt="$VLMCC/training/format_prompts/math.jinja" \
    data.prompt_key=problem \
    data.image_key=images \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.max_pixels=4194304 \
    data.rollout_batch_size=512 \
    data.seed=1 \
    worker.actor.model.model_path="$MODEL" \
    worker.actor.model.freeze_vision_tower=$FREEZE_VIT \
    worker.actor.model.freeze_language_model=$FREEZE_LLM \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.use_torch_compile=false \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="$VLMCC/training/reward_functions/math.py:compute_score" \
    trainer.experiment_name=stage1_${COND}_qwen3vl4b \
    trainer.project_name=VLM-CapCurriculum-RT \
    trainer.total_epochs=16 \
    trainer.n_gpus_per_node=4 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq=6 \
    trainer.save_limit=-1 \
    trainer.save_model_only=false \
    trainer.find_last_checkpoint=true \
    trainer.save_checkpoint_path="$RUN/checkpoints" \
    trainer.logger='["console","wandb"]' \
    ${EXTRA_ARGS:-}
