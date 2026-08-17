#!/bin/bash
# PAPO_FIXED REAL RUN — full paper Eq.2 (C+DE), 60-step trajectory at 8192.
# Uses $SCRATCH/code/PAPO_fixed (B+C fixes + RECOMPUTE_AUG_LOG_PROBS=True), so:
#   GRPO(token) + ref-KL(0.01) + perception-KL(gamma=0.01, MAXIMIZED, full-gradient)
#   + Double Entropy(eta1=eta2=0.03, penalized, ACTIVE via RECOMPUTE) + mask 14px/0.6.
# Deliberate deviation from authors' 2B Table-1 default (which has double entropy OFF) — user
# chose C+DE to match the paper's WRITTEN Eq.2. See PAPO_fixed/README_FIX.md.
#
# gpu_memory_utilization=0.40: the RECOMPUTE masked grad-forward ~doubles update memory (smoke
#   peaked 88.7/95 at 0.45 and still creeping) -> 0.40 gives ~85-87 peak for a safe UNATTENDED run.
# ~26 min/step -> 60 steps ~= 30h -> ~3 resumed 12h slots. Checkpoints 10..60 (optimizer kept).
# Re-run (or afterany-dependency chain) `sbatch papo_2b_8k_papofix_run.sbatch` to auto-resume.
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_fixed
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_papofix_run

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

# auto-resume: on RESUME, skip base val (re-fires per job start) and drop util a touch more for
# an unattended, OOM-safe continuation. Base anchor captured slot 1; final val still fires at 60.
RESUME_ARG=""
LAST_CKPT=$(ls -d "$RUN_DIR"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
    echo "[resume] resuming from $LAST_CKPT"
    RESUME_ARG="trainer.load_checkpoint_path=$LAST_CKPT trainer.val_before_train=false worker.rollout.gpu_memory_utilization=0.38"
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
    worker.rollout.gpu_memory_utilization=0.40 \
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_8k_papofix_run \
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
    trainer.save_freq=10 \
    trainer.save_limit=-1 \
    trainer.save_model_only=false \
    trainer.max_steps=60 \
    $RESUME_ARG \
    "$@"
