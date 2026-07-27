#!/bin/bash
# PAPO_CPURE REAL RUN — the FAITHFUL paper PAPO_G-2B (Table 3: gamma=0.01, NO double entropy,
# ref-KL beta=0.01, mask 0.6), 60-step trajectory at full-chain 8192.
# Code dir = $SCRATCH/code/PAPO_cpure (B+C fixes + RECOMPUTE_AUG_LOG_PROBS=False, entropy OFF via config):
#   GRPO(token) + ref-KL(0.01) + perception-KL(gamma=0.01, MAXIMIZED, real-branch grad) + mask 14px/0.6.
#   NO double entropy (use_aug/ori_entropy_loss NOT overridden -> stay false). NO extra masked forward.
#
# Memory: RECOMPUTE=False -> compute graph == Arm A baseline (no extra masked grad-forward). Arm A ran
#   OK at util 0.60 (peaked ~91/95) and resumed at 0.55. We use 0.55 throughout for an OOM-safe
#   UNATTENDED run. Smoke (papo_2b_8k_cpure_smoke) validates the exact peak first.
# ~60 steps x ~20 min ~= 20h -> ~2 resumed 12h slots. Checkpoints 10..60 (optimizer kept).
# Re-run (or afterany-dependency chain) `sbatch papo_2b_8k_cpure_run.sbatch` to auto-resume.
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_cpure
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_cpure_run

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

# auto-resume: on RESUME, skip base val (re-fires per job start) and drop util a touch more for an
# unattended, OOM-safe continuation. Base anchor captured slot 1; final val still fires at step 60.
RESUME_ARG=""
LAST_CKPT=$(ls -d "$RUN_DIR"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
    echo "[resume] resuming from $LAST_CKPT"
    RESUME_ARG="trainer.load_checkpoint_path=$LAST_CKPT trainer.val_before_train=false worker.rollout.gpu_memory_utilization=0.52"
fi

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
    trainer.experiment_name=papo_qwen3vl2b_8k_cpure_run \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    data.max_prompt_length=4096 \
    algorithm.kl_prcp_coef=0.01 \
    trainer.val_freq=-1 \
    trainer.logger='["console","wandb","file"]' \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints" \
    trainer.save_freq=10 \
    trainer.save_limit=-1 \
    trainer.save_model_only=false \
    trainer.max_steps=60 \
    $RESUME_ARG \
    "$@"
