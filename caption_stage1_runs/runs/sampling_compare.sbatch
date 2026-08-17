#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-sampcmp
#SBATCH --partition=debug
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/sampcmp_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Diagnostic, not the frozen pipeline.
#
# Purpose now: measure the UNCENSORED caption and answer length distributions
# under the revised prompts, so the two caps (D14 and the answer cap) can be set
# from data instead of guessed. Caps are deliberately generous (3072) -- this run
# exists to observe the distribution, not to constrain it.
#
# Sampling settled: untruncated. Prefill REJECTED (it overrides the model's
# natural answering behaviour), so prefill-modes=off.
set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

mkdir -p "$CS1/sampling_compare"

python3 "$CS1/code/sampling_compare.py" \
    --pool "$CS1_POOL" --out "$CS1/sampling_compare" \
    --model "$CS1_MODEL" --snapshot "$CS1_SNAPSHOT" \
    --n-items 10 --g-captions 2 \
    --presets untruncated --prefill-modes off \
    --caption-max-tokens 3072 --answer-max-tokens 3072
