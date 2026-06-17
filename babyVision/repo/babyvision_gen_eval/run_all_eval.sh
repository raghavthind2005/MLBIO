#!/usr/bin/env bash
#
# BabyVision - Run All Evaluations
#
# This script:
# 1. Auto-discovers all models and rounds in generated/
# 2. Runs auto-evaluation for each model/round combination
# 3. Summarizes results with mean/std across rounds
#
# Usage:
#   OPENROUTER_API_KEY=xxx ./run_all_eval.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== Configuration ====================
# Tasks file path (REQUIRED)
TASKS_FILE="${TASKS_FILE:-../data/babyvision_gen_data/meta_data.jsonl}"

# LLM model for evaluation (can be overridden via environment)
LLM_MODEL="${LLM_MODEL:-google/gemini-3-flash-preview}"
OPENROUTER_API_URL="${OPENROUTER_API_URL:-https://openrouter.ai/api/v1/chat/completions}"

# ==================== Help ====================
show_help() {
    cat << EOF
BabyVision - Run All Evaluations

Usage:
    OPENROUTER_API_KEY=xxx ./run_all_eval.sh [options]

Options:
    -h, --help      Show this help message
    --summary-only  Skip evaluation, only generate summary

Description:
    Auto-discovers all models in generated/ directory and runs
    LLM-based evaluation for each model/round combination.

Environment Variables:
    OPENROUTER_API_KEY   API key for OpenRouter (REQUIRED for evaluation)
    OPENROUTER_API_URL   API endpoint URL
    LLM_MODEL            LLM model for evaluation (default: google/gemini-3-flash-preview)
    TASKS_FILE           Path to tasks JSONL file (default: ../data/babyvision_gen_data/meta_data.jsonl)

Output Structure:
    results/
    ├── {model}/
    │   └── {round}/
    │       └── eval.jsonl     # Evaluation results
    ├── summary.txt            # Human-readable summary
    └── summary.json           # Machine-readable summary

Examples:
    # Run full evaluation pipeline
    OPENROUTER_API_KEY=your-key ./run_all_eval.sh

    # Use different evaluation model
    OPENROUTER_API_KEY=your-key LLM_MODEL=google/gemini-2.5-flash ./run_all_eval.sh

    # Custom tasks file
    OPENROUTER_API_KEY=your-key TASKS_FILE=path/to/meta_data.jsonl ./run_all_eval.sh

    # Only generate summary (skip evaluation)
    ./run_all_eval.sh --summary-only

EOF
}

# ==================== Argument Parsing ====================
SUMMARY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --summary-only)
            SUMMARY_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==================== Validation ====================
if [[ "$SUMMARY_ONLY" == "false" ]] && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable is required"
    echo ""
    echo "Usage: OPENROUTER_API_KEY=your-key ./run_all_eval.sh"
    echo "       ./run_all_eval.sh --summary-only  (to skip evaluation)"
    exit 1
fi

# Check tasks file
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "ERROR: Tasks file not found: $TASKS_FILE"
    echo "Please set TASKS_FILE environment variable to the correct path."
    exit 1
fi

echo "========================================"
echo "BabyVision - Run All Evaluations"
echo "========================================"
echo "Tasks File: $TASKS_FILE"
echo "LLM Model: $LLM_MODEL"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ==================== Step 1: Create results directory ====================
mkdir -p results

# ==================== Step 2: Discover models and rounds ====================
echo "--- Discovering Models and Rounds ---"

if [[ ! -d "generated" ]]; then
    echo "ERROR: generated/ directory not found"
    echo "Please run inference first to generate images."
    exit 1
fi

# Find all models in generated folder
MODELS=()
for model_dir in generated/*/; do
    if [[ -d "$model_dir" ]]; then
        model_name=$(basename "$model_dir")
        MODELS+=("$model_name")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No models found in generated/ directory"
    exit 1
fi

echo "Found models: ${MODELS[*]}"
echo ""

# ==================== Step 3: Run evaluations ====================
if [[ "$SUMMARY_ONLY" == "false" ]]; then
    echo "--- Running Auto-Evaluations ---"
    echo ""

    for model in "${MODELS[@]}"; do
        echo "=========================================="
        echo "Model: $model"
        echo "=========================================="

        # Find all rounds for this model
        ROUNDS=()
        for round_dir in "generated/$model"/round*/; do
            if [[ -d "$round_dir" ]]; then
                round_name=$(basename "$round_dir")
                ROUNDS+=("$round_name")
            fi
        done

        if [[ ${#ROUNDS[@]} -eq 0 ]]; then
            echo "WARNING: No rounds found for $model, skipping"
            continue
        fi

        echo "Rounds: ${ROUNDS[*]}"
        echo ""

        for round in "${ROUNDS[@]}"; do
            echo "------------------------------------------"
            echo "Evaluating: $model / $round"
            echo "------------------------------------------"

            OUTPUT_DIR="results/${model}/${round}"
            OUTPUT_FILE="${OUTPUT_DIR}/eval.jsonl"

            # Check if already completed
            if [[ -f "$OUTPUT_FILE" ]]; then
                EXISTING=$(wc -l < "$OUTPUT_FILE")
                echo "Found $EXISTING existing evaluations"
            fi

            python3 scripts/evaluate.py \
                --model-name "$model" \
                --round "$round" \
                --tasks-file "$TASKS_FILE" \
                --llm-model "$LLM_MODEL" \
                --api-key "$OPENROUTER_API_KEY" \
                --api-url "$OPENROUTER_API_URL"

            echo ""
        done
    done
else
    echo "--- Skipping Evaluation (--summary-only) ---"
    echo ""
fi

# ==================== Step 4: Summarize results ====================
echo "--- Summarizing Results ---"
echo ""

python3 scripts/summarize_results.py --tasks-file "$TASKS_FILE"

echo ""
echo "========================================"
echo "All Evaluations Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  results/summary.txt    - Human-readable summary"
echo "  results/summary.json   - Machine-readable summary"
echo ""
echo "To view results:"
echo "  cat results/summary.txt"
echo "  cat results/summary.json | python3 -m json.tool"
