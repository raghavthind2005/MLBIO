#!/bin/bash
# Length/truncation probe — submit with `sbatch probe_len.sbatch`.
# Measures the ACTUAL response-length distribution under model-card sampling at
# max_response_length=8192, to decide (a) whether controlled sampling fixed
# truncation and (b) the real sequence length that drives the training-forward
# memory. Runs the canonical papo_run.sh with only fast/safe overrides:
#   tiny batch (16) + tiny micro-batch (no OOM) + filter off + eager (fast start).
# These are result-NEUTRAL for the length/truncation signal (per-sample).
#SBATCH --account=a0174
#SBATCH --partition=debug
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=papo_probe
#SBATCH --output=/iopsstor/scratch/cscs/%u/runs/papo_run/probe_%j.log

srun --environment="$HOME/toml/verl_easyr1.toml" \
     bash "/iopsstor/scratch/cscs/$USER/code/runs/papo_run.sh" \
     data.rollout_batch_size=16 \
     worker.actor.global_batch_size=16 \
     worker.actor.micro_batch_size_per_device_for_experience=2 \
     worker.actor.micro_batch_size_per_device_for_update=1 \
     data.filter_overlong_prompts=false \
     worker.rollout.enforce_eager=true \
     trainer.max_steps=2 \
     trainer.save_freq=-1
