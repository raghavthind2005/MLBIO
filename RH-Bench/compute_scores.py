"""
Phase 3: Compute final RH-Bench accuracy scores.

Paper: "More Thinking, Less Seeing?" arXiv:2505.21523
Metrics:
  - Reasoning accuracy: % is_correct=True
  - Perception accuracy: % is_correct=True (MC) or hallucination_score >= 3 (free_form)
  - Hallucination threshold: score <= 2 = hallucination, score >= 3 = acceptable
"""
import json, os
from collections import defaultdict

f = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses.json"
if not os.path.exists(f):
    f = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses.json"
    print("Warning: judged file not found, using raw responses (MC accuracy only)")
print(f"Reading: {f}\n")

results = json.load(open(f))
for r in results:
    if r.get("question_type") == "free_from":
        r["question_type"] = "free_form"

stats = defaultdict(lambda: {"correct": 0, "total": 0})
for rec in results:
    key = f"{rec['subset']}_{rec['question_type']}"
    stats[key]["total"] += 1
    if rec.get("is_correct", False):
        stats[key]["correct"] += 1

print("=== RH-Bench Results ===\n")
for task, label in [("reason", "Reasoning"), ("halu", "Perception (Hallucination)")]:
    print(f"--- {label} ---")
    t_total = t_correct = 0
    for qtype in ["multi_choice", "free_form"]:
        k = f"{task}_{qtype}"
        if stats[k]["total"] > 0:
            acc = 100 * stats[k]["correct"] / stats[k]["total"]
            print(f"  {qtype:15s}: {acc:5.1f}%  ({stats[k]['correct']}/{stats[k]['total']})")
            t_total += stats[k]["total"]
            t_correct += stats[k]["correct"]
    if t_total:
        print(f"  {'OVERALL':15s}: {100*t_correct/t_total:5.1f}%  ({t_correct}/{t_total})")
    print()

r_acc = sum(v["correct"] for k, v in stats.items() if k.startswith("reason")) / max(1, sum(v["total"] for k, v in stats.items() if k.startswith("reason")))
h_acc = sum(v["correct"] for k, v in stats.items() if k.startswith("halu")) / max(1, sum(v["total"] for k, v in stats.items() if k.startswith("halu")))
print(f"RH-Bench Point: Reasoning={100*r_acc:.1f}%, Perception={100*h_acc:.1f}%")
print("(RH-AUC requires multiple points across different thinking budgets)")
