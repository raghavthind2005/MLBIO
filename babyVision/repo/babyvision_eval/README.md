# BabyVision MLLM Evaluation

State-of-the-art MLLMs achieve PhD-level language reasoning but struggle with visual tasks that 3-year-olds solve effortlessly. We introduce BabyVision, a benchmark revealing the infancy of AI vision.

## Overview

This package provides tools for:
- Running MLLM inference on BabyVision benchmark tasks
- Using LLM judges to evaluate answer correctness
- Computing detailed scores (overall, type-wise, and subtype-wise)
- Aggregating results across multiple evaluation passes with mean/std

## File Structure

```
BabyVision/                      # Root repository
├── data/
│   ├── babyvision_data.zip      # Benchmark data (need to unzip)
│   ├── babyvision_data/         # Extracted benchmark data
│   │   ├── meta_data.jsonl      # Task definitions
│   │   └── images/              # Task images
│   └── mllm_results.zip         # Example model results
│
└── babyvision_eval/             # This evaluation package
    ├── evaluate_model.py        # Main inference and evaluation script
    ├── compute_score.py         # Score computation and aggregation
    ├── utils.py                 # Utility functions
    ├── run_inference.sh         # Shell wrapper script
    └── results/                 # Output directory
        └── model_results_run_*.json
```

## Configuration

Configuration is set via environment variables.

### Required Variables

| Variable | Description |
|----------|-------------|
| `MODEL_API_KEY` | API key for the model to evaluate |
| `JUDGE_API_KEY` | API key for the judge model |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BASE_URL` | `https://openrouter.ai/api/v1` | Model API endpoint (OpenAI format) |
| `MODEL_NAME` | `google/gemini-3-flash-preview` | Model name to evaluate |
| `JUDGE_BASE_URL` | `https://openrouter.ai/api/v1` | Judge API endpoint |
| `JUDGE_MODEL_NAME` | `openai/gpt-5.2` | Judge model name |
| `TEST_JSON_PATH` | `../data/babyvision_data/meta_data.jsonl` | Path to test JSONL |
| `OUTPUT_DIR` | `./results` | Output directory |
| `NUM_PROCESSES` | `8` | Number of parallel processes |
| `NUM_PASSES` | `3` | Number of evaluation passes |

### Example Setup

```bash
export MODEL_API_KEY="your-model-api-key"
export MODEL_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="google/gemini-3-flash-preview"
export JUDGE_API_KEY="your-judge-api-key"
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_MODEL_NAME="openai/gpt-5.2" # or Qwen-Max 
```

## Usage

### Step 0: Extract Benchmark Data

```bash
unzip data/babyvision_data.zip -d data/
```

This will create `data/babyvision_data/` containing:
- `meta_data.jsonl` - Task definitions
- `images/` - Task images

### Step 1: Run Evaluation

```bash
cd babyvision_eval
export MODEL_API_KEY="your-openrouter-key"
export JUDGE_API_KEY="your-openrouter-key"

bash run_inference.sh
```

Or run Python directly with custom settings:

```bash
python evaluate_model.py \
    --model-api-key $MODEL_API_KEY \
    --model-base-url https://openrouter.ai/api/v1 \
    --model-name google/gemini-2.5-flash \
    --judge-api-key $JUDGE_API_KEY \
    --judge-base-url https://openrouter.ai/api/v1 \
    --judge-model-name openai/gpt-4o \
    --test-json-path ../data/babyvision_data/meta_data.jsonl \
    --output-dir ./results \
    --num-passes 3
```

### Step 2: Compute Scores

```bash
python compute_score.py results/model_results_run_*.json
```

This will output:
- **Overall Average Accuracy**: Mean accuracy with standard deviation
- **Type-wise Average Accuracy**: Breakdown by task type
- **Subtype-wise Average Accuracy**: Detailed breakdown by subtype

## Output Format

### Evaluation Results (`model_results_run_*.json`)

```json
{
  "Id": 123,
  "Question": "...",
  "ModelResult": "The answer is \\boxed{A}",
  "GroundTruth": "A",
  "ExtractedAnswer": "A",
  "LLMJudgeResult": true,
  "Type": "Fine-grained Discrimination",
  "Subtype": "color-recognition"
}
```

### Score Output Example

```
Overall Average Accuracy: 0.7500 ± 0.0200

Type-wise Average Accuracy:
  Fine-grained Discrimination: 0.8000 ± 0.0150
  Visual Tracking: 0.7000 ± 0.0250
  Spatial Perception: 0.7200 ± 0.0180
  Visual Pattern Recognition: 0.7800 ± 0.0120

Subtype-wise Average Accuracy:
  1Fine-grained Discrimination/color-recognition: 0.8500 ± 0.0100
  ...
```

## Example Results

Example model outputs are provided in `../data/mllm_results.zip` for reference and testing `compute_score.py`:

```bash
unzip ../data/mllm_results.zip -d ./example_results/
python compute_score.py ./example_results/*.json
```

## Scoring

- **Accuracy**: `correct / total_tasks`
- LLM judge determines answer correctness by comparing model output to ground truth
- Multiple passes are recommended to account for model variability
- Results are aggregated with mean and standard deviation
