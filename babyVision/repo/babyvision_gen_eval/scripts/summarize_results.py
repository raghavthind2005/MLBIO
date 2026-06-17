#!/usr/bin/env python3
"""
BabyVision Results Summarization Script

Aggregates evaluation results from all models and rounds,
calculates mean and std across rounds for each model.

Results structure:
  results/{model}/{round}/eval.jsonl
  results/summary.txt
  results/summary.json
"""

import json
import math
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"


def load_total_tasks(tasks_file: Path) -> int:
    """Load total number of tasks from tasks.jsonl."""
    if not tasks_file.exists():
        print(f"WARNING: Tasks file not found at {tasks_file}")
        return 0

    count = 0
    with open(tasks_file, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_tasks_info(tasks_file: Path) -> dict:
    """Load task info (type, subtype) from tasks.jsonl."""
    tasks_info = {}
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks_info[task['taskId']] = {
                        'type': task.get('type', 'Unknown'),
                        'subtype': task.get('subtype', 'Unknown')
                    }
    return tasks_info


def discover_models_and_rounds() -> dict:
    """Discover all models and their rounds from results directory."""
    models = {}

    if not RESULTS_DIR.exists():
        return models

    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.is_dir():
            rounds = []
            for round_dir in model_dir.iterdir():
                if round_dir.is_dir() and round_dir.name.startswith('round'):
                    eval_file = round_dir / "eval.jsonl"
                    if eval_file.exists():
                        rounds.append(round_dir.name)
            if rounds:
                models[model_dir.name] = sorted(rounds)

    return models


def load_eval_results(model_name: str, round_name: str) -> list:
    """Load evaluation results from JSONL file."""
    eval_file = RESULTS_DIR / model_name / round_name / "eval.jsonl"
    if not eval_file.exists():
        return []

    results = []
    with open(eval_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_round_stats(results: list, total_tasks: int = None) -> dict:
    """Compute statistics for a single round's results."""
    evaluated = len(results)
    if evaluated == 0:
        return {
            "total": 0, "evaluated": 0, "correct": 0, "incorrect": 0,
            "missing_gen": 0, "api_errors": 0, "accuracy": 0.0
        }

    correct = sum(1 for r in results if r.get("autoEval") is True)
    incorrect = sum(1 for r in results if r.get("autoEval") is False)
    missing_gen = sum(1 for r in results if r.get("missing_generation"))
    api_errors = sum(1 for r in results if r.get("error") is not None and not r.get("missing_generation"))

    if total_tasks is None:
        total_tasks = evaluated

    not_in_results = total_tasks - evaluated
    accuracy = correct / total_tasks if total_tasks > 0 else 0.0

    return {
        "total": total_tasks,
        "evaluated": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "missing_gen": missing_gen,
        "api_errors": api_errors,
        "not_in_results": not_in_results,
        "accuracy": accuracy,
    }


def compute_type_breakdown(results: list, tasks_info: dict = None) -> dict:
    """Compute breakdown by task type and subtype."""
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    subtype_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    seen_task_ids = set()

    for r in results:
        task_id = r.get("taskId")
        seen_task_ids.add(task_id)

        task_type = r.get("type", "Unknown")
        subtype = r.get("subtype", "Unknown")

        type_stats[task_type]["total"] += 1
        subtype_stats[f"{task_type}/{subtype}"]["total"] += 1

        if r.get("autoEval") is True:
            type_stats[task_type]["correct"] += 1
            subtype_stats[f"{task_type}/{subtype}"]["correct"] += 1

    # Add missing tasks as incorrect
    if tasks_info:
        for task_id, info in tasks_info.items():
            if task_id not in seen_task_ids:
                task_type = info.get("type", "Unknown")
                subtype = info.get("subtype", "Unknown")
                type_stats[task_type]["total"] += 1
                subtype_stats[f"{task_type}/{subtype}"]["total"] += 1

    # Calculate accuracy
    for stats in type_stats.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    for stats in subtype_stats.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    return {"by_type": dict(type_stats), "by_subtype": dict(subtype_stats)}


def compute_mean_std(values: list) -> tuple:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0

    n = len(values)
    mean = sum(values) / n

    if n < 2:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    return mean, std


def format_summary_text(model_stats: dict, total_tasks: int = 0) -> str:
    """Format results as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("BabyVision Evaluation Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total tasks in dataset: {total_tasks}")
    lines.append("=" * 80)
    lines.append("")

    # Overall comparison table
    lines.append("-" * 80)
    lines.append("Overall Results (Mean +/- Std across rounds)")
    lines.append("Note: Accuracy = correct / total_tasks (missing generations = incorrect)")
    lines.append("-" * 80)
    lines.append(f"{'Model':<30} {'Rounds':>8} {'Accuracy':>20}")
    lines.append("-" * 80)

    for model_name in sorted(model_stats.keys()):
        data = model_stats[model_name]
        num_rounds = len(data['rounds'])
        mean_acc = data['mean_accuracy']
        std_acc = data['std_accuracy']
        lines.append(f"{model_name:<30} {num_rounds:>8} {mean_acc*100:>8.2f}% +/- {std_acc*100:>5.2f}%")

    lines.append("-" * 80)
    lines.append("")

    # Per-round details
    for model_name in sorted(model_stats.keys()):
        data = model_stats[model_name]

        lines.append("-" * 80)
        lines.append(f"Model: {model_name}")
        lines.append("-" * 80)
        lines.append(f"{'Round':<10} {'Total':>8} {'Eval':>8} {'Correct':>8} {'MissGen':>8} {'Accuracy':>12}")
        lines.append("-" * 80)

        for round_name in sorted(data['rounds'].keys()):
            round_data = data['rounds'][round_name]
            stats = round_data['overall']
            missing_gen = stats.get('missing_gen', 0)
            evaluated = stats.get('evaluated', stats['total'])
            lines.append(f"{round_name:<10} {stats['total']:>8} {evaluated:>8} {stats['correct']:>8} {missing_gen:>8} {stats['accuracy']*100:>11.2f}%")

        lines.append("-" * 80)
        lines.append(f"{'AVERAGE':<10} {'':<8} {'':<8} {'':<8} {'':<8} {data['mean_accuracy']*100:>8.2f}% +/- {data['std_accuracy']*100:.2f}%")
        lines.append("-" * 80)
        lines.append("")

    # Type breakdown
    lines.append("-" * 80)
    lines.append("Breakdown by Task Type (averaged across rounds)")
    lines.append("-" * 80)

    for model_name in sorted(model_stats.keys()):
        data = model_stats[model_name]
        type_accuracies = data.get('type_accuracies', {})

        if type_accuracies:
            lines.append(f"\n{model_name}:")
            lines.append(f"{'Type':<45} {'Accuracy':>15}")
            lines.append("-" * 60)

            for task_type in sorted(type_accuracies.keys()):
                mean_acc, std_acc = type_accuracies[task_type]
                lines.append(f"{task_type:<45} {mean_acc*100:>6.2f}% +/- {std_acc*100:>5.2f}%")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='BabyVision Results Summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_results.py --tasks-file data/babyvision_gen_data/meta_data.jsonl
"""
    )

    parser.add_argument('--tasks-file', type=str, required=True, help='Path to tasks JSONL file (meta_data.jsonl)')

    args = parser.parse_args()

    # Resolve tasks file path
    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"[ERROR] Tasks file not found: {tasks_file}")
        sys.exit(1)

    print("Summarizing evaluation results...")
    print(f"Tasks file: {tasks_file}")

    total_tasks = load_total_tasks(tasks_file)
    print(f"Total tasks in dataset: {total_tasks}")

    tasks_info = load_tasks_info(tasks_file)
    print(f"Loaded {len(tasks_info)} task info records")

    models_rounds = discover_models_and_rounds()

    if not models_rounds:
        print("No evaluation results found in results/ directory")
        print("Expected structure: results/{model}/{round}/eval.jsonl")
        return

    print(f"Found {len(models_rounds)} models")
    for model, rounds in models_rounds.items():
        print(f"  {model}: {len(rounds)} rounds ({', '.join(rounds)})")

    # Compute statistics
    model_stats = {}

    for model_name, rounds in models_rounds.items():
        round_accuracies = []
        round_data = {}
        type_accuracies_by_round = defaultdict(list)

        for round_name in rounds:
            results = load_eval_results(model_name, round_name)
            if results:
                overall = compute_round_stats(results, total_tasks=total_tasks)
                breakdown = compute_type_breakdown(results, tasks_info=tasks_info)

                round_data[round_name] = {
                    "overall": overall,
                    "breakdown": breakdown,
                }

                round_accuracies.append(overall['accuracy'])

                for task_type, stats in breakdown['by_type'].items():
                    type_accuracies_by_round[task_type].append(stats['accuracy'])

        mean_acc, std_acc = compute_mean_std(round_accuracies)

        type_accuracies = {}
        for task_type, accs in type_accuracies_by_round.items():
            type_accuracies[task_type] = compute_mean_std(accs)

        model_stats[model_name] = {
            "rounds": round_data,
            "round_accuracies": round_accuracies,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "type_accuracies": type_accuracies,
        }

    # Generate outputs
    summary_text = format_summary_text(model_stats, total_tasks=total_tasks)

    print("")
    print(summary_text)

    # Save text summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_file = RESULTS_DIR / "summary.txt"
    with open(txt_file, "w") as f:
        f.write(summary_text)
    print(f"Saved text summary to: {txt_file}")

    # Save JSON summary
    json_output = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "note": "Accuracy = correct / total_tasks. Missing generations are treated as incorrect.",
        "models": {},
    }

    for model_name, data in model_stats.items():
        json_output["models"][model_name] = {
            "num_rounds": len(data['rounds']),
            "mean_accuracy": data['mean_accuracy'],
            "std_accuracy": data['std_accuracy'],
            "round_accuracies": data['round_accuracies'],
            "type_accuracies": {k: {"mean": v[0], "std": v[1]} for k, v in data['type_accuracies'].items()},
            "rounds": data['rounds'],
        }

    json_file = RESULTS_DIR / "summary.json"
    with open(json_file, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON summary to: {json_file}")


if __name__ == "__main__":
    main()
