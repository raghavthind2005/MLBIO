#!/bin/bash
# PAPO full training run on Qwen3-VL-4B-Thinking (MLBIO) — THE canonical run script.
#
# The MIRROR DIAGNOSTIC is this EXACT script + `trainer.max_steps=2 trainer.save_freq=-1`
# (appended via "$@"). So the diagnostic is byte-identical to the real run except for
# step count + checkpointing => its signal (truncation / length / speed / memory / reward)
# transfers directly to sign-off. Per-knob provenance + sign-off: runs/RESULTS_PAPO.md
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO
MODEL=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking
RUN_DIR=$SCRATCH/runs/papo_run

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PAPO_DIR:${PYTHONPATH:-}       # PAPO's verl (conv-patched) must win
export HF_HOME=$SCRATCH/hf_cache                   # pre-downloaded ViRL39K / MMK12
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_memory_usage_threshold=0.98
export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR
# NB: never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — breaks vLLM CuMemAllocator.

mkdir -p "$RUN_DIR"
cd "$PAPO_DIR"

python3 -m verl.trainer.main \
    config=examples/configs/config_grpo_papo.yaml \
    data.train_files=PAPOGalaxy/PAPO_ViRL39K_train \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    data.rollout_batch_size=384 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.max_pixels=1003520 \
    data.min_pixels=200704 \
    data.filter_overlong_prompts=true \
    data.format_prompt=examples/format_prompt/math_perception.jinja \
    worker.actor.model.model_path="$MODEL" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.global_batch_size=128 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.ref.fsdp.enable_cpu_offload=false \
    worker.rollout.n=5 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.95 \
    worker.rollout.top_k=20 \
    worker.rollout.enforce_eager=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.gpu_memory_utilization=0.85 \
    worker.rollout.disable_tqdm=true \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    algorithm.use_kl_prcp=true \
    algorithm.kl_prcp_coef=0.01 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.max_steps=60 \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name=papo_mlbio \
    trainer.experiment_name=papo_qwen3vl4b_thinking \
    trainer.save_freq=6 \
    trainer.save_limit=-1 \
    trainer.save_model_only=false \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints" \
    trainer.val_freq=-1 \
    trainer.val_before_train=true \
    "$@"
