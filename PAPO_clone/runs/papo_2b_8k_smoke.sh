#!/bin/bash
# PAPO 2B-Thinking 8K SMOKE — memory re-check at full-chain response length.
# Vs the passing 2048 smoke, changes are:
#   [SCIENCE]        data.max_response_length=8192            full reasoning chains (intended)
#   [RESULT-NEUTRAL] micro_batch_for_update=1                 mem lever; global_batch=128 unchanged
#   [RESULT-NEUTRAL] micro_batch_for_experience=4             mem lever; per-forward tokens == 2048 run
#   [REQUIRED]       max_num_batched_tokens=16384             vLLM asserts >= prompt+response(=12288)
# The micro-batches are sized so per-forward token count == the 2048 run that fit (peak 85/95).
# Smoke caps: max_steps=2, save_freq=-1, val_before_train=false (memory-focused; T_val@8192
# will come from the real run's base val — val is vLLM-only gen and won't OOM).
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_clone
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_smoke

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export PYTHONPATH=$PAPO_DIR:${PYTHONPATH:-}
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR
# never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — breaks vLLM CuMemAllocator.

mkdir -p "$RUN_DIR"
cd "$PAPO_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/configs/config_grpo_papo.yaml \
    data.train_files=PAPOGalaxy/PAPO_ViRL39K_train \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    data.rollout_batch_size=384 \
    data.max_response_length=8192 \
    data.format_prompt=examples/format_prompt/math_perception.jinja \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_8k_smoke \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    data.max_prompt_length=4096 \
    algorithm.kl_prcp_coef=0.01 \
    trainer.val_freq=-1 \
    trainer.logger='["console","wandb","file"]' \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints" \
    trainer.max_steps=2 \
    trainer.save_freq=-1 \
    trainer.val_before_train=false \
    "$@"
