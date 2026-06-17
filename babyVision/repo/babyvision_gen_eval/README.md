# BabyVision Generation Evaluation

State-of-the-art MLLMs achieve PhD-level language reasoning but struggle with visual tasks that 3-year-olds solve effortlessly. We introduce BabyVision, a benchmark revealing the infancy of AI vision.

## Overview

This package provides tools for:
- Running image generation inference on BabyVision benchmark tasks
- Using LLM models to evaluate generated images against ground truth
- Computing detailed scores (overall, type-wise, and subtype-wise)
- Aggregating results across multiple evaluation rounds with mean/std

## File Structure

```
BabyVision/                      # Root repository
├── data/
│   ├── babyvision_gen_data.zip  # Benchmark data (need to unzip)
│   └── babyvision_gen_data/     # Extracted benchmark data
│       ├── meta_data.jsonl      # Task definitions (280 tasks)
│       ├── images/              # Input puzzle images
│       └── answerImages/        # Ground truth annotated images
│
└── babyvision_gen_eval/         # This evaluation package
    ├── scripts/
    │   ├── inference.py         # Generate annotated images via API
    │   ├── evaluate.py          # Auto-evaluate generated images using LLM judge
    │   └── summarize_results.py # Aggregate results and compute statistics
    ├── inference.sh             # Shell wrapper for inference
    ├── run_all_eval.sh          # Run full evaluation pipeline
    ├── generated/               # Generated images output
    │   └── {model_name}/
    │       └── round{N}/
    │           ├── images/      # Generated annotated images
    │           └── results.jsonl
    └── results/                 # Evaluation results
        ├── {model_name}/
        │   └── round{N}/
        │       └── eval.jsonl   # Per-task evaluation results
        ├── summary.txt          # Human-readable summary
        └── summary.json         # Machine-readable summary
```

## Configuration

Configuration is set via environment variables.

### Required Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter (required for inference and evaluation) |

### Optional Variables

#### Inference Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `google/gemini-3-pro-image-preview` | Model for image generation |
| `MODEL_NAME` | `nanobanana-pro` | Output directory name under `generated/` |
| `ROUNDS` | `3` | Number of generation rounds |
| `NUM_WORKERS` | `4` | Number of parallel workers |
| `REQUEST_DELAY` | `1.0` | Delay between API requests (seconds) |

#### Evaluation Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `google/gemini-3-flash-preview` | LLM model for evaluation |
| `TASKS_FILE` | `../data/babyvision_gen_data/meta_data.jsonl` | Path to tasks JSONL file |

### Example Setup

```bash
export OPENROUTER_API_KEY="your-api-key"
export MODEL_NAME="my-model"
export ROUNDS=3
export TASKS_FILE="../data/babyvision_gen_data/meta_data.jsonl"
```

## Usage

### Step 0: Extract Benchmark Data

Before running any evaluation, extract the benchmark data:

```bash
unzip data/babyvision_gen_data.zip -d data/
```

This will create `data/babyvision_gen_data/` containing:
- `meta_data.jsonl` - Task definitions
- `images/` - Input puzzle images
- `answerImages/` - Ground truth annotated images

### Step 1: Run Inference

Generate annotated images for the benchmark tasks:

```bash
cd babyvision_gen_eval
export OPENROUTER_API_KEY="your-api-key"

# Run inference (default: 3 rounds)
./inference.sh

# Or with custom settings
MODEL_NAME=gpt-image MODEL=openai/gpt-4o ROUNDS=3 ./inference.sh
```

Generated images will be saved to:
```
generated/{MODEL_NAME}/round{N}/images/
```

### Step 2: Run Evaluation

Evaluate all generated models:

```bash
export OPENROUTER_API_KEY="your-api-key"
./run_all_eval.sh
```

Or evaluate a specific model/round:

```bash
python scripts/evaluate.py \
    --model-name nanobanana-pro \
    --round round1 \
    --tasks-file ../data/babyvision_gen_data/meta_data.jsonl \
    --api-key $OPENROUTER_API_KEY
```

### Step 3: View Results

Results are saved to `results/`:

```bash
# View human-readable summary
cat results/summary.txt

# View JSON summary
cat results/summary.json | python3 -m json.tool
```

### Generate Summary Only

If evaluation is already complete, generate summary without re-running evaluation:

```bash
./run_all_eval.sh --summary-only
```

Or directly:

```bash
python scripts/summarize_results.py --tasks-file ../data/babyvision_gen_data/meta_data.jsonl
```

## Custom Inference (Using Your Own Code)

If you want to use your own inference code instead of the provided `inference.py`, you can do so as long as the generated images follow the expected format.

### Expected Directory Structure

```
generated/{model_name}/round{N}/images/
```

### Image Naming Convention

For each task in `meta_data.jsonl`, the generated image should be named:

```
{image_uuid}_task{taskId}.png
```

Where:
- `{image_uuid}` is the filename (without extension) from the task's `image` field
- `{taskId}` is the task ID from the task's `taskId` field

**Example:** For a task with:
```json
{
  "taskId": 445,
  "image": "images/1323d501-85a7-4a9e-abe4-2f5b7e22e458.jpg",
  ...
}
```

The generated image should be saved as:
```
generated/my-model/round1/images/1323d501-85a7-4a9e-abe4-2f5b7e22e458_task445.png
```

### Supported Image Formats

- PNG (recommended)
- JPG/JPEG

### After Generating Images

Once your images are in the correct location, run the evaluation:

```bash
export OPENROUTER_API_KEY="your-api-key"
./run_all_eval.sh
```

The evaluation script will automatically discover all models and rounds in the `generated/` directory.

## Output Format

### Evaluation Results (`results/{model}/{round}/eval.jsonl`)

```json
{
  "taskId": 445,
  "type": "Fine-grained Discrimination",
  "subtype": "Find the different",
  "autoEval": true,
  "error": null,
  "response": "True"
}
```

### Summary (`results/summary.json`)

```json
{
  "generated_at": "2024-01-10T12:00:00",
  "total_tasks": 280,
  "models": {
    "nanobanana-pro": {
      "num_rounds": 3,
      "mean_accuracy": 0.75,
      "std_accuracy": 0.02,
      "round_accuracies": [0.74, 0.75, 0.76],
      "type_accuracies": {
        "Fine-grained Discrimination": {"mean": 0.80, "std": 0.01},
        "Visual Tracking": {"mean": 0.70, "std": 0.02}
      }
    }
  }
}
```

## Scoring

- **Accuracy**: `correct / total_tasks`
- Missing generated images are counted as incorrect
- Results are aggregated across multiple rounds to compute mean and standard deviation
