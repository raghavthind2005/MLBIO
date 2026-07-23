#!/bin/bash
# PAPO 2B-Thinking SMOKE — dot-identical mirror of PAPO's
#   examples/papo_grpo/qwen3_vl_2b_grpo_papo.sh
# Every python arg is verbatim from their script; the ONLY changes are:
#   [OPS]  non-training: model path, offline/HF/W&B env, logging, PYTHONPATH, ckpt path
#   [SMOKE] training-neutral caps: max_steps=2, save_freq=-1   <-- removed for the real run
# Base val + final val are KEPT (their default val_before_train=true) to (a) exercise the
# val path and (b) measure T_val for the real-run length estimate.
# We run PAPO ONLY (config use_kl_prcp=true) — NOT the baseline grpo script.
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_clone                 # [OPS] fresh clone @1263a29 + 2 container-compat patches
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking   # [OPS] local dir (their script used the HF hub id)
RUN_DIR=$SCRATCH/runs/papo_2b_smoke               # [OPS]

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98            # theirs
export PYTHONPATH=$PAPO_DIR:${PYTHONPATH:-}       # [OPS] patched verl must win
export HF_HOME=$SCRATCH/hf_cache                  # [OPS]
export HF_HUB_OFFLINE=1                           # [OPS]
export WANDB_MODE=offline                         # [OPS]
export WANDB_DIR=$RUN_DIR                         # [OPS]
# [OPS] never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — breaks vLLM CuMemAllocator.

mkdir -p "$RUN_DIR"
cd "$PAPO_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/configs/config_grpo_papo.yaml \
    data.train_files=PAPOGalaxy/PAPO_ViRL39K_train \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    data.rollout_batch_size=384 \
    data.format_prompt=examples/format_prompt/math_perception.jinja \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_thinking_smoke \
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
    "$@"
