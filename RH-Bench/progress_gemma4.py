"""
Live progress monitor for Gemma-4 RH-Bench run.
Run from login node at any time:
  python3 progress_gemma4.py
  watch -n 30 'python3 /iopsstor/scratch/cscs/raghavthind/code/rh-bench/progress_gemma4.py'
"""
import json, os, time
from collections import defaultdict

BASE   = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results"
INF_F  = f"{BASE}/vlm_responses_gemma4.json"
JUD_F  = f"{BASE}/judged_responses_gemma4.json"

def load(f):
    if not os.path.exists(f): return []
    try:
        return json.load(open(f))
    except:
        return []

print(f"\n{'='*55}")
print(f"  RH-Bench Gemma-4-31B Progress  [{time.strftime('%H:%M:%S')}]")
print(f"{'='*55}")

# Inference progress
inf = load(INF_F)
if not inf:
    print("\n[Inference] Not started yet.")
else:
    think = sum(1 for r in inf if r.get("has_thinking"))
    avg_t = sum(r.get("elapsed_s", 0) for r in inf) / len(inf)
    avg_w = sum(r.get("thinking_words", 0) for r in inf) / max(1, think)
    halu  = [r for r in inf if r["subset"] == "halu"]
    reas  = [r for r in inf if r["subset"] == "reason"]
    eta   = (1000 - len(inf)) * avg_t / 60

    print(f"\n[Inference]  {len(inf)}/1000 done  ({len(halu)} halu, {len(reas)} reason)")
    print(f"  Thinking ON:    {think}/{len(inf)} ({100*think/len(inf):.0f}%)")
    print(f"  Avg think words: {avg_w:.0f}")
    print(f"  Avg time/sample: {avg_t:.1f}s")
    print(f"  ETA remaining:   {eta:.0f} min")

# Judge progress
jud = load(JUD_F)
if not jud:
    print("\n[Judge]      Not started yet.")
else:
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for rec in jud:
        if rec.get("question_type") == "free_from":
            rec["question_type"] = "free_form"
        key = f"{rec['subset']}_{rec['question_type']}"
        stats[key]["total"] += 1
        if rec.get("is_correct", False):
            stats[key]["correct"] += 1

    print(f"\n[Judge]      {len(jud)}/{len(inf) or 1000} judged")
    for task, label in [("reason", "Reasoning"), ("halu", "Perception")]:
        t_tot = t_cor = 0
        for qtype in ["multi_choice", "free_form"]:
            k = f"{task}_{qtype}"
            if stats[k]["total"]:
                t_tot += stats[k]["total"]
                t_cor += stats[k]["correct"]
        if t_tot:
            print(f"  {label}: {100*t_cor/t_tot:.1f}% ({t_cor}/{t_tot})")

print(f"\n{'='*55}\n")
