#!/usr/bin/env python3
"""
BabyVision GRADING — simple, once and for all.

One method for every question type: show the judge the model's FINAL answer
(answer_text) together with the question and the correct answer, and ask whether
they match in SUBSTANCE, ignoring formatting. No \\boxed{} extraction, no letter
regex, no string-normalization rules — those are what created the mess and the
`\\boxed{Answer}` format penalty. The judge reads what the model actually wrote,
exactly like a human TA grading against a key.

Writes `grade` (bool) + `grade_raw` into results_run{N}_graded.jsonl for every
condition. Self-validates on CHOICE questions (gold letter is known) so we can
SEE the judge agrees with ground truth rather than assume it.

Needs the Qwen3 judge server running (same one run_judge.py uses, port 30001).

Usage:
  python grade.py --base /iopsstor/.../babyvision --port 30001
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# (key, dir, passes)
CONDITIONS = [
    ("a0",       "results_a0_nothink",        [1]),
    ("standard", "results_standard",          [1, 2, 3]),
    ("a3",       "results_a3_forced_long",    [1]),
    ("a3v1",     "results_a3_forced_long_v1", [1]),
    ("b1",       "results_b1_reinject",       [1]),
    ("b2",       "results_b2_noreinject",     [1]),
]

GRADE_PROMPT = """You are grading one answer to a visual puzzle. Decide if the model's answer is correct.

Question:
{question}

Correct answer:
{gold}

Model's answer:
{answer}

Mark CORRECT if the model's FINAL answer means the same as the correct answer, even if it is
phrased or formatted differently (for example "(B)" vs "B", "option B" vs "B",
"Row 2, Column 3" vs "second row third column", "8-5=3" vs "8-5=3"), ignoring any surrounding
explanation. Mark INCORRECT if the final answer is a different value, is missing, or contradicts
the correct answer. If the model states multiple answers, grade its FINAL one.

Answer with exactly one word: CORRECT or INCORRECT."""

N_CONCURRENT = 16
MAX_ANSWER_CHARS = 8000


def model_answer(rec: dict) -> str:
    """The model's final committed answer, format-agnostic.

    Normal case: the model's final answer text (answer_text); fall back to the
    tail of the thinking trace only if the final answer is empty.

    Two-turn (B1/B2) special case: if turn 2 ran out of budget *while still
    thinking* (finish_reason=='length' AND the thinking channel never closed, so
    thinking_trace==''), then answer_text holds cut-off, often degenerate-looping
    turn-2 reasoning with NO conclusion — not an answer. The model's standing
    committed answer there is its turn-1 answer (the 2-turn protocol already
    defines extracted_answer with this same T1 fallback in run_infer_b.py). Grade
    that instead. Inert for single-turn conditions (A*), which have no
    turn1_answer_text field, so their grading is byte-identical to before."""
    t2_unconcluded = (
        rec.get("turn1_answer_text") is not None
        and rec.get("finish_reason") == "length"
        and not (rec.get("thinking_trace") or "").strip()
    )
    if t2_unconcluded:
        a = (rec.get("turn1_answer_text") or "").strip()
        if not a:
            a = (rec.get("turn1_thinking") or "")[-1500:].strip()
        return a[:MAX_ANSWER_CHARS]

    a = (rec.get("answer_text") or "").strip()
    if not a:
        a = (rec.get("thinking_trace") or "")[-1500:].strip()
    return a[:MAX_ANSWER_CHARS]


def call_grade(url, model, question, gold, answer):
    prompt = GRADE_PROMPT.format(question=question, gold=gold,
                                 answer=answer if answer else "(no answer given)")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 16,
        "temperature": 0.0,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
    # order matters: "incorrect" contains "correct"
    if "incorrect" in raw:
        return False, raw
    if "correct" in raw:
        return True, raw
    if "true" in raw:
        return True, raw
    if "false" in raw:
        return False, raw
    return False, raw   # unparseable → conservative


def grade_record(rec, url, model, lock, counter):
    if "error" in rec:
        rec["grade"] = False
        rec["grade_raw"] = "SKIP_ERROR"
        return rec
    if rec.get("grade") is not None:          # resume
        return rec
    try:
        ok, raw = call_grade(url, model, rec.get("question_sent", ""),
                             rec.get("gt_answer", ""), model_answer(rec))
        rec["grade"] = ok
        rec["grade_raw"] = raw
    except Exception as e:
        rec["grade"] = False
        rec["grade_raw"] = f"ERROR:{e}"
    with lock:
        counter[0] += 1
        n, tot = counter
        print(f"  [{n:4d}/{tot}] id={rec.get('taskId'):>5} "
              f"{'✓' if rec['grade'] else '✗'} gt={str(rec.get('gt_answer',''))[:12]:12s} "
              f"raw={rec['grade_raw'][:12]!r}", flush=True)
    return rec


# ── choice self-validation (gold letter known) ──
def letter_of(s):
    if s is None:
        return None
    s = str(s)
    m = re.findall(r"\(([A-Ha-h])\)", s) or re.findall(r"\b([A-Ha-h])\b", s)
    return m[-1].upper() if m else None


def grade_dir(base, key, dirname, passes, url, model, val):
    cond_dir = base / dirname
    for pi in passes:
        src = cond_dir / f"results_run{pi}.jsonl"
        if not src.exists():
            continue
        out = cond_dir / f"results_run{pi}_graded.jsonl"
        recs = [json.loads(l) for l in open(src) if l.strip()]
        print(f"\n{dirname} pass{pi}: grading {len(recs)} records → {out.name}")
        lock, counter = threading.Lock(), [0, len(recs)]
        with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
            graded = [f.result() for f in
                      [pool.submit(grade_record, r, url, model, lock, counter) for r in recs]]
        with open(out, "w") as f:
            for r in graded:
                f.write(json.dumps(r) + "\n")

        valid = [r for r in graded if "error" not in r]
        acc = sum(1 for r in valid if r.get("grade") is True) / len(valid) if valid else 0
        if pi == 1:
            # self-validation on choice
            choice = [r for r in valid if r.get("ansType") == "choice"]
            agree = disagree = 0
            for r in choice:
                gl = letter_of(r.get("gt_answer"))
                ml = letter_of(model_answer(r))
                det = (ml is not None and ml == gl)
                if det == (r.get("grade") is True):
                    agree += 1
                else:
                    disagree += 1
            val[key] = {"acc": acc, "n": len(valid),
                        "choice_n": len(choice), "agree": agree, "disagree": disagree}
        print(f"  acc(pass{pi}) = {acc*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    model = args.model or requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"Judge: {model}")

    base = Path(args.base)
    val = {}
    for key, dirname, passes in CONDITIONS:
        if (base / dirname).exists():
            grade_dir(base, key, dirname, passes, url, model, val)

    print("\n" + "=" * 70)
    print("  SUMMARY  (grade = judge on the model's final answer, format-agnostic)")
    print("=" * 70)
    print(f"  {'condition':<26}{'n':>5}{'acc':>8}{'choice agree/disagree':>24}")
    for key, dirname, _ in CONDITIONS:
        if key in val:
            v = val[key]
            print(f"  {dirname:<26}{v['n']:>5}{v['acc']*100:>7.1f}%"
                  f"{v['agree']:>15}/{v['disagree']:<8}")
    print("""
  CHOICE self-validation: 'agree' = grade matches the known gold letter; 'disagree'
  = they differ (inspect — should be tiny). High agreement ⇒ the judge is faithful to
  ground truth and we can trust it on the free-form blanks too.""")


if __name__ == "__main__":
    main()
