#!/usr/bin/env bash
#
# BabyVision Inference Script
#
# Generate annotated images for BabyVision tasks using image generation models.
# This script wraps scripts/inference.py with convenient defaults and configuration.
#
# Usage:
#   ./inference.sh                              # Use defaults
#   ./inference.sh --model-name my-model        # Custom model name
#   OPENROUTER_API_KEY=xxx ./inference.sh       # Set API key via env
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# All settings can be overridden via environment variables

# Data paths (relative to repo root)
DATA_ROOT="${DATA_ROOT:-../data/babyvision_gen_data}"
JSONL_PATH="${JSONL_PATH:-../data/babyvision_gen_data/meta_data.jsonl}"

# Output directory - model name will be appended
MODEL_NAME="${MODEL_NAME:-nanobanana-pro}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/generated/${MODEL_NAME}}"

# Generation settings
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"

# API settings (API key MUST be set via environment variable)
OPENROUTER_API_URL="${OPENROUTER_API_URL:-https://openrouter.ai/api/v1/chat/completions}"
MODEL="${MODEL:-google/gemini-3-pro-image-preview}"

# Performance settings
NUM_WORKERS="${NUM_WORKERS:-4}"
REQUEST_DELAY="${REQUEST_DELAY:-1.0}"

# ==================== Help ====================
show_help() {
    cat << EOF
BabyVision Image Generation Inference

Usage:
    ./inference.sh [options]
    OPENROUTER_API_KEY=xxx ./inference.sh

Options:
    -h, --help          Show this help message
    --model-name NAME   Set output model name (default: nanobanana-pro)

Description:
    Generates annotated images for BabyVision visual reasoning tasks.
    Results are saved to generated/{model_name}/round{N}/images/

Environment Variables:
    OPENROUTER_API_KEY   API key for OpenRouter (REQUIRED)
    OPENROUTER_API_URL   API endpoint URL
    MODEL                Model to use for generation
    MODEL_NAME           Output directory name under generated/
    DATA_ROOT            Path to data directory
    JSONL_PATH           Path to task metadata JSONL
    OUTPUT_DIR           Custom output directory
    ROUNDS               Number of generation rounds (default: 3)
    SEED                 Random seed (default: 42)
    NUM_WORKERS          Parallel workers (default: 4)
    REQUEST_DELAY        Delay between API requests (default: 1.0s)

Examples:
    # Basic usage with API key
    OPENROUTER_API_KEY=your-key ./inference.sh

    # Custom model name and rounds
    OPENROUTER_API_KEY=your-key MODEL_NAME=gpt-image ROUNDS=5 ./inference.sh

    # Use different generation model
    OPENROUTER_API_KEY=your-key MODEL=openai/gpt-4o ./inference.sh

Output Structure:
    generated/{model_name}/
    ├── round1/
    │   ├── images/          # Generated images
    │   └── results.jsonl    # Metadata with image paths
    ├── round2/
    └── round3/

EOF
}

# ==================== Argument Parsing ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --model-name)
            MODEL_NAME="$2"
            OUTPUT_DIR="${SCRIPT_DIR}/generated/${MODEL_NAME}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==================== Validation ====================
# Check API key
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable is required"
    echo ""
    echo "Usage: OPENROUTER_API_KEY=your-key ./inference.sh"
    exit 1
fi

# Check data paths
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: Data root not found: $DATA_ROOT"
    echo "Please ensure the data is extracted to the correct location."
    exit 1
fi

if [[ ! -f "$JSONL_PATH" ]]; then
    echo "ERROR: JSONL file not found: $JSONL_PATH"
    exit 1
fi

TASK_COUNT=$(wc -l < "$JSONL_PATH")

# ==================== Display Configuration ====================
echo "=============================================="
echo "  BabyVision Image Generation Inference"
echo "=============================================="
echo "Configuration:"
echo "  Data Root:     $DATA_ROOT"
echo "  JSONL:         $JSONL_PATH"
echo "  Output:        $OUTPUT_DIR"
echo "  Tasks:         $TASK_COUNT"
echo "  Rounds:        $ROUNDS"
echo "  Seed:          $SEED"
echo "  Model:         $MODEL"
echo "  API URL:       $OPENROUTER_API_URL"
echo "  Workers:       $NUM_WORKERS"
echo "  Request Delay: ${REQUEST_DELAY}s"
echo "=============================================="
echo ""

# ==================== Run Inference ====================
START_TIME=$(date +%s)

python3 "${SCRIPT_DIR}/scripts/inference.py" \
    --data-root "$DATA_ROOT" \
    --jsonl "$JSONL_PATH" \
    --output "$OUTPUT_DIR" \
    --rounds "$ROUNDS" \
    --seed "$SEED" \
    --api-key "$OPENROUTER_API_KEY" \
    --api-url "$OPENROUTER_API_URL" \
    --model "$MODEL" \
    --num-workers "$NUM_WORKERS" \
    --request-delay "$REQUEST_DELAY"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "          Inference Complete"
echo "=============================================="
echo "Total time: ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "Output directory: ${OUTPUT_DIR}"
for i in $(seq 1 "$ROUNDS"); do
    echo "  - round${i}/: images/ + results.jsonl"
done
echo "=============================================="
