#!/bin/bash
# =============================================================================
# Stage 1 (Visual Perception) RLVR — CONDITION 1: full fine-tune (LLM + ViT)
# = MAXIMALLY paper-faithful reproduction of the paper's Stage 1.
#
# Every RESULT-AFFECTING knob = the paper's original value (max_pixels=4194304,
# max_response_length=2048, rollout_batch_size=512, global_batch_size=128,
# total_epochs=16, n=5, lr=1e-6, KL low_var_kl/1e-2, reward, data).
#
# Deviations are RESULT-NEUTRAL or forced by hardware (none change the trained model):
#   [NEUTRAL] vision patch-embed Conv3d -> matmul (monkey-patch in EasyR1); bit-identical
#             output (maxdiff=0). The aarch64 cuDNN Conv3d was ~3e5x too slow. See RESULTS.md §7.1.
#   [NEUTRAL] micro_batch_size_per_device 16/32 -> 4/8 (gradient-accumulation chunking; the
#             gradient is identical. paper's 16 OOMs at response=2048 on 96GB GH200 vs 141GB H200).
#   [NEUTRAL] PYTORCH_ALLOC_CONF=expandable_segments:True (fragmentation), use_torch_compile=false.
#   [HW]      n_gpus_per_node 8 -> 4 (sharding; global/rollout batch held -> same optimization).
#   [ENG]     reward_type via REWARD_TYPE module attr (engine-version drift; same "batch" behavior).
#   [OBS]     in-run val disabled (eval checkpoints offline); frequent full resumable checkpoints.
# =============================================================================
#SBATCH --job-name=stage1_full
#SBATCH --account=a0174
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/runs/stage1_full/slurm-%j.out

set -euo pipefail

EASYR1=/iopsstor/scratch/cscs/raghavthind/code/EasyR1
VLMCC=/iopsstor/scratch/cscs/raghavthind/code/VLM-CapCurriculum
DATA=/capstor/store/cscs/swissai/a0174/datasets/VLM-CapCurriculum-Perception-Data
MODEL=/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct
RUN=/iopsstor/scratch/cscs/raghavthind/runs/stage1_full
mkdir -p "$RUN"

export PYTHONPATH=$EASYR1:${PYTHONPATH:-}
export WANDB_MODE=offline
export WANDB_DIR=$RUN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

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
    worker.actor.model.model_path="$MODEL" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.model.freeze_language_model=false \
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
    trainer.experiment_name=stage1_full_qwen3vl4b \
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
