#!/bin/bash
# Runtime SMOKE for PAPO on Qwen3-VL-4B-Thinking (MLBIO).
#
# PURPOSE: prove the full GRPO+PAPO loop runs end-to-end on 4xGH200 for 2 steps
# (vLLM rollout -> reward -> PAPO masked-image forward -> FSDP update, no save),
# and read the key signals (format-reward fires, truncation rate, kl_prcp active,
# step-time, no OOM) BEFORE committing a full run.
#
# MIRROR PRINCIPLE: every result- and memory-shaping knob below is IDENTICAL to the
# intended full run (all paper/repo-faithful). The ONLY differences vs the full run
# are the last block: max_steps=2, save off, val off. Flip those to promote to full.
#
# Run inside the EasyR1/PAPO container (has verl deps). Needs >=4 GPUs, ~1h budget.
set -x

# ---- paths ----
SCRATCH=/iopsstor/scratch/cscs/$USER
PAPO_DIR=$SCRATCH/code/PAPO
MODEL=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking
RUN_DIR=$SCRATCH/runs/papo_smoke

# ---- env ----
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PAPO_DIR:${PYTHONPATH:-}        # PAPO's verl (with conv patch) must win
export HF_HOME=$SCRATCH/hf_cache                    # pre-downloaded ViRL39K / MMK12
export HF_HUB_OFFLINE=1                             # use the cache, don't re-fetch
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_memory_usage_threshold=0.98
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline                           # offline; logs land in WANDB_DIR
export WANDB_DIR=$RUN_DIR

mkdir -p "$RUN_DIR"
cd "$PAPO_DIR"

python3 -m verl.trainer.main \
    config=examples/configs/config_grpo_papo.yaml \
    data.train_files=PAPOGalaxy/PAPO_ViRL39K_train \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    data.rollout_batch_size=384 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.format_prompt=examples/format_prompt/math_perception.jinja \
    worker.actor.model.model_path="$MODEL" \
    worker.actor.global_batch_size=128 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=examples/reward_function/qwen3_vl_think.py:compute_score \
    algorithm.use_kl_prcp=true \
    algorithm.kl_prcp_coef=0.01 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=papo_mlbio \
    trainer.experiment_name=smoke_qwen3vl4b_thinking_papo \
    trainer.max_steps=2 \
    trainer.save_freq=-1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false
