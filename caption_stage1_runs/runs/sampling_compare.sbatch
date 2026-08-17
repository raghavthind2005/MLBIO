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
# Sampling settled: untruncated (D23) -- that is a CORRECTNESS constraint
# (truncated sampling draws from p-tilde != p, biasing every D-hat) and this run
# does not reopen it. Prefill REJECTED (it overrides the model's natural
# answering behaviour), so prefill-modes=off.
#
# BOTH presets are swept under D33 because D23's *safety* argument -- that
# off-card sampling is also better behaved (2B: rep 0.035 untruncated vs 0.450
# model-card, 13x) -- was measured on the 2B and does not automatically transfer
# to the 4B backbone. We re-establish it rather than inherit it.
#
# n-items raised 10 -> 30: the 2B run yielded only 20 answers and its median sat
# pinned at the 3072 cap (censored), which cannot set a cap.
set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

mkdir -p "$CS1/sampling_compare"

python3 "$CS1/code/sampling_compare.py" \
    --pool "$CS1_POOL" --out "$CS1/sampling_compare" \
    --model "$CS1_MODEL" --snapshot "$CS1_SNAPSHOT" \
    --n-items 30 --g-captions 2 \
    --presets untruncated,model_card --prefill-modes off \
    --caption-max-tokens 3072 --answer-max-tokens 3072
