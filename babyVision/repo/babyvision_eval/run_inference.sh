#!/usr/bin/env bash
#
# BabyVision MLLM Evaluation Script
#
# Evaluates multimodal large language models on the BabyVision benchmark.
#
# Usage:
#   1. Set the environment variables below
#   2. Run: bash run_inference.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# Model to evaluate
export MODEL_API_KEY="${MODEL_API_KEY:-your-openrouter-api-key}"
export MODEL_BASE_URL="${MODEL_BASE_URL:-https://openrouter.ai/api/v1}"
export MODEL_NAME="${MODEL_NAME:-google/gemini-3-flash}"

# Judge model for evaluation
export JUDGE_API_KEY="${JUDGE_API_KEY:-your-openrouter-api-key}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-openai/gpt-5.2}"

# Data paths
export TEST_JSON_PATH="${TEST_JSON_PATH:-../data/babyvision_data/meta_data.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-./results}"

# Processing settings
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export NUM_PASSES="${NUM_PASSES:-3}"

# ==================== Validation ====================

if [[ ! -f "$TEST_JSON_PATH" ]]; then
    echo "ERROR: Test JSON file not found: $TEST_JSON_PATH"
    echo "Please unzip the data first: unzip ../data/babyvision_data.zip -d ../data/"
    exit 1
fi

# ==================== Display Configuration ====================
echo "========================================"
echo "BabyVision MLLM Evaluation"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Judge: $JUDGE_MODEL_NAME"
echo "Test Data: $TEST_JSON_PATH"
echo "Output: $OUTPUT_DIR"
echo "Processes: $NUM_PROCESSES"
echo "Passes: $NUM_PASSES"
echo "========================================"
echo ""

# ==================== Run Evaluation ====================
python3 "${SCRIPT_DIR}/evaluate_model.py" \
    --model-api-key "$MODEL_API_KEY" \
    --model-base-url "$MODEL_BASE_URL" \
    --model-name "$MODEL_NAME" \
    --judge-api-key "$JUDGE_API_KEY" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model-name "$JUDGE_MODEL_NAME" \
    --test-json-path "$TEST_JSON_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-processes "$NUM_PROCESSES" \
    --num-passes "$NUM_PASSES"

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To compute scores:"
echo "  python compute_score.py ${OUTPUT_DIR}/model_results_run_*.json"
