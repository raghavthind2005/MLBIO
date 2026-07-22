#!/bin/bash
# FINAL comprehensive verification — submit with `sbatch ckpt_verify.sbatch`.
# Validates everything no prior smoke did, on the canonical papo_run.sh (tiny batch,
# separate test dir): (1) checkpoint saving (save_freq=1 -> shards on disk at step
# 1 & 2, ready ~20min in); (2) file+wandb logging; (3) the VALIDATION path — the
# unavoidable end-of-fit _validate() over MMK12 (same code as val_before_train).
# We skip the *initial* val here (val_before_train=false) so the smoke runs ONE
# val, not two. ~1h total (checkpoints verifiable at ~20min, before the val).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_ckpt
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_ckpt_test/ckpt_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_run.sh" \
     data.rollout_batch_size=16 \
     worker.actor.global_batch_size=16 \
     worker.actor.micro_batch_size_per_device_for_experience=2 \
     worker.actor.micro_batch_size_per_device_for_update=1 \
     data.filter_overlong_prompts=false \
     worker.rollout.enforce_eager=true \
     trainer.max_steps=2 \
     trainer.save_freq=1 \
     trainer.val_before_train=false \
     trainer.experiment_name=papo_ckpt_test \
     trainer.save_checkpoint_path=/iopsstor/scratch/cscs/$USER/runs/papo_ckpt_test/checkpoints
