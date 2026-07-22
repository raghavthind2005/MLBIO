#!/bin/bash
# Checkpoint + logging verification — submit with `sbatch ckpt_verify.sbatch`.
# The ONE thing no smoke tested: save_freq>0 actually writes usable checkpoints
# (the study data) + the file/wandb logs. Runs the canonical papo_run.sh with
# save ON at every step, tiny batch, to a SEPARATE test dir (won't touch the real
# run's checkpoints). ~15 min. Verify BEFORE the 10h run.
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:40:00
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
     trainer.experiment_name=papo_ckpt_test \
     trainer.save_checkpoint_path=/iopsstor/scratch/cscs/$USER/runs/papo_ckpt_test/checkpoints
