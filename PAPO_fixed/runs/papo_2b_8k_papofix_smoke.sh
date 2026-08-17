#!/bin/bash
# PAPO_FIXED smoke — the REAL PAPO objective (Eq. 2), memory+timing re-check.
# vs the baseline 8k smoke, this uses the FIXED repo ($SCRATCH/code/PAPO_fixed):
#   - B fix: perception/entropy/sft losses now BACKPROPPED (was orphaned).
#   - C fix: aux metrics averaged (kl_prcp_loss etc. now reliable, not last-sample noise).
#   - RECOMPUTE_AUG_LOG_PROBS=True: extra masked grad-forward -> full-gradient perception KL +
#     ACTIVE double-entropy eta_2 (dot-identical to paper Eq. 2). ~doubles update memory.
# Double entropy explicitly ON (algorithm.use_aug/ori_entropy_loss=true).
# gpu_memory_utilization lowered 0.6 -> 0.45 to absorb the extra forward (result-neutral).
# Smoke caps: max_steps=2, save_freq=-1, val_before_train=false.
# PASS = no OOM + advances to step 2 + (now-reliable) kl_prcp_loss is being MAXIMIZED (magnitude
#        rises / stays high, NOT decaying to 0) = direct proof PAPO is finally training.
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_fixed                 # FIXED repo (B+C + RECOMPUTE=True)
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_papofix_smoke

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export PYTHONPATH=$PAPO_DIR:${PYTHONPATH:-}       # fixed verl must win
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
    worker.rollout.gpu_memory_utilization=0.45 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_papofix_smoke \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    data.max_prompt_length=4096 \
    algorithm.kl_prcp_coef=0.01 \
    algorithm.use_aug_entropy_loss=true \
    algorithm.use_ori_entropy_loss=true \
    trainer.val_freq=-1 \
    trainer.logger='["console","wandb","file"]' \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints" \
    trainer.max_steps=2 \
    trainer.save_freq=-1 \
    trainer.val_before_train=false \
    "$@"
