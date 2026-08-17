#!/bin/bash
#SBATCH --account=a0174
#SBATCH --job-name=cs1-score
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/raghavthind/caption_stage1/logs/score_%j.log
#SBATCH --environment=/users/raghavthind/toml/verl_easyr1.toml

# Grade an answers file (Job A). CPU-only and cheap, which is the point:
# generation is the expensive, hard-to-repeat half, so scoring is a SEPARATE
# job that can be re-run any number of times against the same artifacts without
# regenerating a single token.
#
# Usage:
#   sbatch score.sbatch <answers.jsonl> <out.json> [label]

set -euo pipefail
source /iopsstor/scratch/cscs/raghavthind/caption_stage1/runs/_env.sh

ANSWERS=${1:?usage: score.sbatch <answers.jsonl> <out.json> [label]}
OUTJSON=${2:?usage: score.sbatch <answers.jsonl> <out.json> [label]}
LABEL=${3:-a}

echo "answers: $ANSWERS"
echo "out    : $OUTJSON"
echo

python3 "$CS1/code/pilot_score.py" \
    --pool "$CS1_POOL" \
    --answers "$ANSWERS" \
    --out "$OUTJSON" \
    --label "$LABEL"
