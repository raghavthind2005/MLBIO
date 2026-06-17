#!/usr/bin/env python3
"""
BabyVision Auto-Evaluation Script

Evaluates generated images from any model against ground truth using LLM-based evaluation.
Supports multiple rounds for each model with resume capability.

Usage:
    python evaluate.py --model-name gpt-image-1.5 --round round1
    python evaluate.py --model-name nanobanana --round round2
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, Lock

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

# Paths
GENERATED_DIR = BASE_DIR / "generated"
RESULTS_DIR = BASE_DIR / "results"

# Default API Configuration
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "google/gemini-3-flash-preview"

# Rate limiting defaults
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 30.0

# Global locks for thread safety
request_lock = Lock()
last_request_time = [0.0]


# ============================================================================
# Image Loading
# ============================================================================

def load_image_as_base64(image_path: str) -> Tuple[str, str]:
    """Load image and return base64 string with mime type."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/png')

    with open(image_path, 'rb') as f:
        image_data = f.read()

    return base64.b64encode(image_data).decode('utf-8'), mime_type


def find_generated_image(task: Dict, model_name: str, round_name: str) -> Optional[str]:
    """Find generated image path for a task."""
    image_path = task.get('image', '')
    task_id = task.get('taskId')
    uuid = Path(image_path).stem

    gen_dir = GENERATED_DIR / model_name / round_name / "images"

    patterns = [
        f"{uuid}_task{task_id}.png",
        f"{uuid}_task{task_id}.jpg",
        f"{uuid}.png",
        f"{uuid}.jpg",
    ]

    for pattern in patterns:
        gen_path = gen_dir / pattern
        if gen_path.exists():
            return str(gen_path)

    return None


# ============================================================================
# Evaluation Prompts
# ============================================================================

def build_evaluation_prompt(task: Dict) -> str:
    """Build type-specific evaluation prompt."""
    task_type = task.get('type', '')
    subtype = task.get('subtype', '')
    generation_prompt = task.get('generationPrompt', '')

    header = f"""You are evaluating an AI-generated image for a visual reasoning task.

TASK TYPE: {task_type}
SUBTYPE: {subtype}
GENERATION INSTRUCTION: "{generation_prompt}"

You are provided with THREE images:
- **Image 1 (Input)**: The original question/puzzle image
- **Image 2 (Ground Truth)**: The CORRECT answer showing what the result SHOULD look like
- **Image 3 (Generated)**: The AI-generated result to be evaluated

Compare Image 3 (Generated) with Image 2 (Ground Truth) to determine if they show the SAME answer.
"""

    criteria = get_type_criteria(task_type, subtype)

    footer = """

DECISION RULES:
- TRUE: Generated image shows the EXACT SAME answer as Ground Truth
- FALSE: Generated shows a DIFFERENT answer, NO answer, or UNCLEAR answer

IMPORTANT:
- Focus ONLY on whether the ANSWER matches, ignore style differences
- A marking on a DIFFERENT element/option = FALSE
- A path taking a DIFFERENT route = FALSE
- A DIFFERENT number or character = FALSE
- Missing required answer = FALSE

Respond with ONLY one word: "True" or "False"
"""

    return header + criteria + footer


def get_type_criteria(task_type: str, subtype: str) -> str:
    """Get evaluation criteria based on task type."""

    if task_type == "Fine-grained Discrimination":
        if subtype == "Find the different":
            return """
TASK: Find the unique/different element among many similar elements.
CRITERIA: Is the circle on the SAME grid position as ground truth?
- Circle on DIFFERENT element = FALSE
- No circle visible = FALSE"""

        elif subtype == "Find the same":
            return """
TASK: Find identical elements or matching figures.
CRITERIA: Are ALL marked elements the same as in ground truth?
- Circles on DIFFERENT elements = FALSE
- Missing circles = FALSE"""

        elif subtype == "Find the shadow":
            return """
TASK: Find the shadow/silhouette that matches the colored figure.
CRITERIA: Is the SAME option circled?"""

        elif subtype in ["Find the same components", "2D Pattern Completion", "Pattern and Color Completion"]:
            return """
TASK: Find/select the correct option.
CRITERIA: Is the SAME option circled/selected?"""

        elif subtype in ["Count Same Patterns", "Count Clusters"]:
            return """
TASK: Count patterns or fill in numbers.
CRITERIA: Do the markings/numbers match ground truth exactly?"""

    elif task_type == "Visual Tracking":
        if subtype == "Maze":
            return """
TASK: Draw a path through the maze.
CRITERIA: Does the path follow the EXACT SAME route as ground truth?
- Different route = FALSE
- No visible path = FALSE"""

        elif subtype == "Connect the lines":
            return """
TASK: Trace a line following the continuous path.
CRITERIA: Does the traced line follow the SAME path as ground truth?"""

        elif subtype == "Metro map":
            return """
TASK: Draw the shortest path between metro stations.
CRITERIA: Does the path follow the EXACT SAME route as ground truth?"""

        elif subtype == "Recognize numbers and letters":
            return """
TASK: Fill in letters/numbers in blanks.
CRITERIA: Are the EXACT SAME characters filled in each blank?"""

    elif task_type == "Spatial Perception":
        if subtype in ["3D Views", "3D Cube Unfold", "Paper Folding", "3D Pattern Completion"]:
            return """
TASK: Select the correct option for spatial reasoning.
CRITERIA: Is the SAME option circled?"""

        elif subtype == "Count 3D blocks":
            return """
TASK: Count cubes in a 3D structure.
CRITERIA: Is the EXACT SAME number written?"""

    elif task_type == "Visual Pattern Recognition":
        return """
TASK: Identify pattern and select correct option.
CRITERIA: Is the SAME option circled?"""

    return """
CRITERIA: Does the answer match ground truth exactly?
- Different answer = FALSE
- Missing answer = FALSE"""


# ============================================================================
# API Evaluation
# ============================================================================

def evaluate_with_api(
    task: Dict,
    input_path: str,
    gt_path: str,
    gen_path: str,
    llm_model: str,
    api_key: str,
    api_url: str,
    request_delay: float,
    max_retries: int,
    retry_delay: float,
) -> Tuple[Optional[bool], str, str]:
    """Evaluate using LLM API. Returns (result, error, raw_response)."""

    try:
        input_b64, input_mime = load_image_as_base64(input_path)
        gt_b64, gt_mime = load_image_as_base64(gt_path)
        gen_b64, gen_mime = load_image_as_base64(gen_path)
    except Exception as e:
        return None, f"Error loading images: {e}", ""

    prompt = build_evaluation_prompt(task)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://babyvision-eval.local",
        "X-Title": "BabyVision Auto-Evaluation"
    }

    payload = {
        "model": llm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "**Image 1 (Input):**"},
                {"type": "image_url", "image_url": {"url": f"data:{input_mime};base64,{input_b64}"}},
                {"type": "text", "text": "\n**Image 2 (Ground Truth):**"},
                {"type": "image_url", "image_url": {"url": f"data:{gt_mime};base64,{gt_b64}"}},
                {"type": "text", "text": "\n**Image 3 (Generated):**"},
                {"type": "image_url", "image_url": {"url": f"data:{gen_mime};base64,{gen_b64}"}},
                {"type": "text", "text": "\n" + prompt}
            ]
        }],
        "max_tokens": 200,
        "temperature": 0.0
    }

    # Rate limiting
    with request_lock:
        elapsed = time.time() - last_request_time[0]
        if elapsed < request_delay:
            time.sleep(request_delay - elapsed)
        last_request_time[0] = time.time()

    # Make request with retries
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=180)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                print(f"  Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text[:200]}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None, error_msg, ""

            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            if 'true' in content.lower():
                return True, "", content
            elif 'false' in content.lower():
                return False, "", content
            else:
                return None, f"Unexpected response: {content}", content

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None, "Request timeout", ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None, str(e), ""

    return None, "Max retries exceeded", ""


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def load_tasks(tasks_file: Path) -> list:
    """Load tasks from JSONL file."""
    tasks = []
    with open(tasks_file, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line.strip()))
    return tasks


def load_existing_results(output_path: Path) -> Dict[int, Dict]:
    """Load existing results to support resume."""
    results = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task_id = data.get('taskId')
                    if task_id is not None:
                        results[task_id] = data
                except json.JSONDecodeError:
                    continue
    return results


def evaluate_task(
    task: Dict,
    model_name: str,
    round_name: str,
    llm_model: str,
    api_key: str,
    api_url: str,
    request_delay: float,
    max_retries: int,
    retry_delay: float,
    idx: int,
    total: int,
    semaphore: Semaphore,
    output_lock: Lock,
    output_file,
    data_dir: Path,
) -> Dict:
    """Evaluate a single task."""
    task_id = task.get('taskId')
    task_type = task.get('type', '')
    subtype = task.get('subtype', '')

    with semaphore:
        print(f"[{idx+1}/{total}] Task {task_id} ({task_type}/{subtype})...")

        input_path = str(data_dir / task.get('image', ''))
        gt_path = str(data_dir / task.get('answerImage', ''))
        gen_path = find_generated_image(task, model_name, round_name)

        # Check paths exist
        if not os.path.exists(input_path):
            error = f"Input image not found: {input_path}"
            print(f"  ERROR: {error}")
            result_data = {'taskId': task_id, 'autoEval': None, 'error': error}
            with output_lock:
                output_file.write(json.dumps(result_data) + '\n')
                output_file.flush()
            return result_data

        if not os.path.exists(gt_path):
            error = f"Ground truth not found: {gt_path}"
            print(f"  ERROR: {error}")
            result_data = {'taskId': task_id, 'autoEval': None, 'error': error}
            with output_lock:
                output_file.write(json.dumps(result_data) + '\n')
                output_file.flush()
            return result_data

        if not gen_path:
            error = f"Generated image not found for task {task_id}"
            print(f"  MISSING: {error} (marked as False)")
            result_data = {
                'taskId': task_id,
                'type': task_type,
                'subtype': subtype,
                'autoEval': False,
                'error': error,
                'response': None,
                'missing_generation': True
            }
            with output_lock:
                output_file.write(json.dumps(result_data) + '\n')
                output_file.flush()
            return result_data

        # Evaluate
        result, error, response = evaluate_with_api(
            task, input_path, gt_path, gen_path, llm_model,
            api_key, api_url, request_delay, max_retries, retry_delay
        )

        result_data = {
            'taskId': task_id,
            'type': task_type,
            'subtype': subtype,
            'autoEval': result,
            'error': error if error else None,
            'response': response if response else None
        }

        if error:
            print(f"  ERROR: {error}")
        else:
            status = "CORRECT" if result else "INCORRECT"
            print(f"  {status}")

        with output_lock:
            output_file.write(json.dumps(result_data) + '\n')
            output_file.flush()

        return result_data


def print_summary(results: Dict[int, Dict], total_tasks: int = None):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    correct = sum(1 for r in results.values() if r.get('autoEval') == True)
    incorrect = sum(1 for r in results.values() if r.get('autoEval') == False)
    missing_gen = sum(1 for r in results.values() if r.get('missing_generation'))
    api_errors = sum(1 for r in results.values() if r.get('error') and not r.get('missing_generation'))

    if total_tasks is None:
        total_tasks = len(results)

    missing_in_results = total_tasks - len(results)

    print(f"Total tasks in dataset: {total_tasks}")
    print(f"Total evaluated: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Incorrect (evaluated): {incorrect - missing_gen}")
    print(f"Missing generations (marked False): {missing_gen}")
    print(f"API errors: {api_errors}")
    if missing_in_results > 0:
        print(f"Tasks not in results: {missing_in_results}")

    print(f"\nAccuracy (correct / total_tasks): {correct/total_tasks*100:.1f}%")

    # Breakdown by type
    print("\nBreakdown by Type/Subtype:")
    type_stats = {}
    for r in results.values():
        key = f"{r.get('type', 'Unknown')}/{r.get('subtype', 'Unknown')}"
        if key not in type_stats:
            type_stats[key] = {'correct': 0, 'incorrect': 0, 'error': 0}
        if r.get('error'):
            type_stats[key]['error'] += 1
        elif r.get('autoEval') == True:
            type_stats[key]['correct'] += 1
        elif r.get('autoEval') == False:
            type_stats[key]['incorrect'] += 1

    for key, stats in sorted(type_stats.items()):
        total = stats['correct'] + stats['incorrect']
        if total > 0:
            acc = stats['correct'] / total * 100
            print(f"  {key}: {stats['correct']}/{total} ({acc:.1f}%)")


def get_available_models() -> list:
    """Get list of available models from generated folder."""
    models = []
    if GENERATED_DIR.exists():
        for d in GENERATED_DIR.iterdir():
            if d.is_dir():
                models.append(d.name)
    return sorted(models)


def get_available_rounds(model_name: str) -> list:
    """Get list of available rounds for a model."""
    rounds = []
    model_dir = GENERATED_DIR / model_name
    if model_dir.exists():
        for d in model_dir.iterdir():
            if d.is_dir() and d.name.startswith('round'):
                rounds.append(d.name)
    return sorted(rounds)


def main():
    parser = argparse.ArgumentParser(
        description='BabyVision Auto-Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific model and round
  python evaluate.py --model-name gpt-image-1.5 --round round1 --tasks-file data/babyvision_gen_data/meta_data.jsonl

  # Use custom LLM model
  python evaluate.py --model-name nanobanana --round round1 --tasks-file data/meta_data.jsonl --llm-model google/gemini-2.5-flash

Environment Variables:
  OPENROUTER_API_KEY    API key for OpenRouter (required if not using --api-key)
  OPENROUTER_API_URL    API URL (default: https://openrouter.ai/api/v1/chat/completions)
  LLM_MODEL             LLM model for evaluation (default: google/gemini-3-flash-preview)
"""
    )

    parser.add_argument('--model-name', type=str, required=True, help='Name of the image generation model to evaluate')
    parser.add_argument('--round', type=str, required=True, help='Round name (e.g., round1, round2, round3)')
    parser.add_argument('--tasks-file', type=str, required=True, help='Path to tasks JSONL file (meta_data.jsonl)')
    parser.add_argument('--llm-model', type=str, default=None, help='LLM model for evaluation')
    parser.add_argument('--api-key', type=str, default=None, help='OpenRouter API key (or set OPENROUTER_API_KEY)')
    parser.add_argument('--api-url', type=str, default=None, help='OpenRouter API URL')
    parser.add_argument('--max-concurrent', type=int, default=DEFAULT_MAX_CONCURRENT, help='Max concurrent API requests')
    parser.add_argument('--request-delay', type=float, default=DEFAULT_REQUEST_DELAY, help='Delay between requests')
    parser.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES, help='Max retries per request')
    parser.add_argument('--retry-delay', type=float, default=DEFAULT_RETRY_DELAY, help='Delay between retries')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (for debugging)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of cases to evaluate')

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] API key required. Use --api-key or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Get API URL and LLM model from args or environment
    api_url = args.api_url or os.environ.get("OPENROUTER_API_URL", DEFAULT_API_URL)
    llm_model = args.llm_model or os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)

    # Resolve tasks file path
    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"[ERROR] Tasks file not found: {tasks_file}")
        sys.exit(1)

    print("BabyVision Auto-Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Round: {args.round}")
    print(f"Tasks File: {tasks_file}")
    print(f"LLM: {llm_model}")
    print(f"Sequential: {args.sequential}")
    print()

    # Check if model and round exist
    model_dir = GENERATED_DIR / args.model_name / args.round / "images"
    if not model_dir.exists():
        print(f"ERROR: Directory not found: {model_dir}")
        print(f"\nAvailable models: {get_available_models()}")
        if (GENERATED_DIR / args.model_name).exists():
            print(f"Available rounds for {args.model_name}: {get_available_rounds(args.model_name)}")
        sys.exit(1)

    # Setup output directory
    output_dir = RESULTS_DIR / args.model_name / args.round
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval.jsonl"

    # Determine data directory from tasks file location
    data_dir = tasks_file.parent

    # Load tasks and existing results
    tasks = load_tasks(tasks_file)
    print(f"Loaded {len(tasks)} tasks")

    existing_results = load_existing_results(output_path)
    print(f"Found {len(existing_results)} existing evaluations")

    # Filter pending tasks
    pending_tasks = [t for t in tasks if t['taskId'] not in existing_results]

    # Check generated images
    missing = sum(1 for t in pending_tasks if not find_generated_image(t, args.model_name, args.round))
    print(f"Generated images found: {len(pending_tasks) - missing}/{len(pending_tasks)}")

    if args.limit:
        pending_tasks = pending_tasks[:args.limit]

    print(f"Pending evaluations: {len(pending_tasks)}")
    print()

    if not pending_tasks:
        print("All tasks already evaluated!")
    else:
        output_lock = Lock()
        semaphore = Semaphore(args.max_concurrent if not args.sequential else 1)
        start_time = time.time()

        with open(output_path, 'a') as output_file:
            if args.sequential:
                for idx, task in enumerate(pending_tasks):
                    evaluate_task(
                        task, args.model_name, args.round, llm_model,
                        api_key, api_url, args.request_delay, args.max_retries, args.retry_delay,
                        idx, len(pending_tasks), semaphore, output_lock, output_file, data_dir
                    )
            else:
                with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
                    futures = [
                        executor.submit(
                            evaluate_task, task, args.model_name, args.round, llm_model,
                            api_key, api_url, args.request_delay, args.max_retries, args.retry_delay,
                            idx, len(pending_tasks), semaphore, output_lock, output_file, data_dir
                        )
                        for idx, task in enumerate(pending_tasks)
                    ]
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Task failed: {e}")

        print(f"\nCompleted in {time.time() - start_time:.1f}s")

    # Print summary
    all_results = load_existing_results(output_path)
    print_summary(all_results, total_tasks=len(tasks))


if __name__ == '__main__':
    main()
