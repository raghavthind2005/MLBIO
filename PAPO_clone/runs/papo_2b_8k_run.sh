#!/bin/bash
# PAPO 2B-Thinking REAL RUN — 60-step mechanistic trajectory at full-chain 8192.
# Identical recipe to the PASSING 8k smoke; smoke caps removed, checkpointing + base/final val ON.
#
# max_steps=60 ONLY changes the number of iterations. LR is flat (warmup_style=constant,
# warmup_ratio=0) and kl_prcp_schedule=fixed, so NO schedule depends on training_steps ->
# stopping at 60 and RESUMING later (just raise max_steps) is EQUIVALENT to running straight.
#
# ~60 steps x ~20 min ~= 20h -> ~2 resumed 12h slots. Checkpoints at 10/20/30/40/50/60 (+ base
# model) with optimizer kept (save_model_only=false) so resume/continue is exact.
# Re-run `sbatch papo_2b_8k_run.sbatch` to auto-resume from the latest checkpoint.
set -x

SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO_clone
MODEL_PATH=$SCRATCH/models/Qwen3-VL-2B-Thinking
RUN_DIR=$SCRATCH/runs/papo_2b_8k_run

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

# auto-resume: verl does NOT auto-load; point at the latest global_step_* so a resubmit
# CONTINUES (dataloader + optimizer + global_step restored) instead of restarting at 0.
RESUME_ARG=""
LAST_CKPT=$(ls -d "$RUN_DIR"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
    echo "[resume] resuming from $LAST_CKPT"
    # skip base val on RESUME: val_before_train (ray_trainer.py:715) re-fires on every job
    # start and would waste ~2h re-validating the resumed ckpt. Base anchor is already
    # captured in slot 1; the final val (ray_trainer.py:853) still fires at step 60.
    RESUME_ARG="trainer.load_checkpoint_path=$LAST_CKPT trainer.val_before_train=false"
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
    worker.actor.global_batch_size=128 \
    trainer.experiment_name=papo_qwen3vl2b_8k_de_run \
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
