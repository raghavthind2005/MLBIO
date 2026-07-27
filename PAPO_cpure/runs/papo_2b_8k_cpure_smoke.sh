#!/bin/bash
# PAPO_CPURE memory+correctness smoke on debug. Submit:
#   sbatch $SCRATCH/code/runs/papo_2b_8k_cpure_smoke.sbatch
# Validates the EXACT real-run config (util 0.55, 8192, RECOMPUTE=False, entropy OFF) for 2 steps:
#   (1) no OOM (peak reserved < 95); (2) perception KL MAXIMIZED (actor/kl_prcp_loss magnitude rises,
#   not decays); (3) config dump: use_aug/ori_entropy_loss=false, kl_prcp_coef=0.01, use_kl_prcp=true,
#   and NO actor/ori_entropy_loss metric. ~30-45 min (no extra masked forward -> faster than C+DE).
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_cpure
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_cpure_smoke

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
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_8k_cpure_smoke \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    data.max_prompt_length=4096 \
    algorithm.kl_prcp_coef=0.01 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb","file"]' \
    trainer.save_freq=-1 \
    trainer.max_steps=2 \
    "$@"
