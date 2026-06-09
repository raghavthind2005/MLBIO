"""
Post-hoc: Fix GT letter extraction bugs and recompute multi-choice accuracy.

Use this after Phase 1 completes if you want to check MC accuracy before Phase 2 runs.
Also normalizes the 'free_from' typo in the dataset.
"""
import json, re
from collections import defaultdict


def parse_choices(question_text):
    choices = {}
    for m in re.finditer(r'\(([A-E])\)\s*(.*?)(?=\n\s*\([A-E]\)|\Z)', question_text, re.DOTALL):
        choices[m.group(1).upper()] = m.group(2).strip()
    return choices


def find_gt_letter(answer, question_text):
    if re.match(r'^[A-E]$', answer.strip(), re.IGNORECASE):
        return answer.strip().upper()
    choices = parse_choices(question_text)
    if not choices:
        return answer.strip().upper()[:1] if answer.strip() else None
    ans_norm = re.sub(r'\s+', '', answer.strip().lower().replace('\\', ''))
    for letter, text in choices.items():
        t_norm = re.sub(r'\s+', '', text.lower().replace('\\', ''))
        if ans_norm == t_norm or (len(ans_norm) > 2 and ans_norm in t_norm) or (len(t_norm) > 2 and t_norm in ans_norm):
            return letter
    for letter, text in choices.items():
        if answer.strip().lower() in text.lower():
            return letter
    return None


def extract_pred_letter(text):
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not clean.strip():
        return None
    for p in [
        r'\*\*[Aa]nswer[:\s]+\(?([A-E])\)?',
        r'[Aa]nswer[:\s]+\(?([A-E])\)?',
        r'[Cc]orrect\s+(?:answer|option|choice)[:\s]+\(?([A-E])\)?',
        r'[Tt]he\s+(?:correct\s+)?answer\s+is\s+\(?([A-E])\)?',
        r'[Oo]ption\s+\(?([A-E])\)?\s+is\s+correct',
        r'[Mm]y\s+answer\s+is\s+\(?([A-E])\)?',
    ]:
        m = re.search(p, clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    lines = [l.strip() for l in clean.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m = re.match(r'^["\']?\(?([A-E])\)?["\']?[.\s]*$', last, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r'(?:therefore|so|thus|hence)[,\s]+\(?([A-E])\)?', last, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    all_letters = re.findall(r'\b([A-E])\b', clean, re.IGNORECASE)
    return all_letters[-1].upper() if all_letters else None


INPUT  = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses.json"
OUTPUT = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses_fixed.json"

d = json.load(open(INPUT))
print(f"Loaded {len(d)} responses")

typo_fixed = sum(1 for r in d if r.get("question_type") == "free_from")
for rec in d:
    if rec.get("question_type") == "free_from":
        rec["question_type"] = "free_form"
print(f"Normalized 'free_from' typo: {typo_fixed} entries")

for rec in d:
    if rec["question_type"] != "multi_choice":
        continue
    rec["gt_letter"] = find_gt_letter(rec["gt_answer"], rec["question"])
    rec["pred_letter"] = extract_pred_letter(rec["raw_response"])

json.dump(d, open(OUTPUT, "w"), indent=2)

mc = [x for x in d if x["question_type"] == "multi_choice"]
no_pred = sum(1 for x in mc if not x.get("pred_letter"))
no_gt = sum(1 for x in mc if not x.get("gt_letter"))
print(f"\nMC total: {len(mc)} | missing pred: {no_pred} | missing GT: {no_gt}")

stats = defaultdict(lambda: {"correct": 0, "total": 0})
for rec in mc:
    pred, gt = rec.get("pred_letter"), rec.get("gt_letter")
    key = f"{rec['subset']}_mc"
    stats[key]["total"] += 1
    if pred and gt and pred == gt:
        stats[key]["correct"] += 1

print("\n=== Multi-choice accuracy (letter match, pre-judge) ===")
for k, v in sorted(stats.items()):
    acc = 100 * v["correct"] / v["total"] if v["total"] else 0
    print(f"  {k:25s}: {acc:.1f}%  ({v['correct']}/{v['total']})")
print(f"\nFixed responses saved to {OUTPUT}")
print("Run compute_scores.py after judge completes for full results.")
