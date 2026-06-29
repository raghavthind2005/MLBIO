#!/usr/bin/env python3
"""BabyVision ERROR-MODE analysis — WHAT does the model do wrong, and is it fixable?

For the stable-wrong items (P(correct)=0 over all 7 draws — the categorically failed
ones), classify the PRIMARY error in the model's standard-pass reasoning into:

  PERCEPTION  the model's *description of the image* is wrong (miscounts, misses /
              invents elements, wrong positions/colors/shapes/relations). Error is
              in seeing. → fix is visual (resolution, tiling, encoder).
  REASONING   the description is accurate but the conclusion is wrong (logic /
              procedure error from correct observations). → fix is reasoning.
  INDECISION  no committed answer; contradicts itself / hedges.
  OTHER       none of the above.

This both diagnoses the fix target AND tests the perception-bound thesis: if nearly
all failures are PERCEPTION (the model's own description is already wrong), the
bottleneck is seeing, not thinking.

CAVEAT: the classifier is the Qwen3 judge reading the model's *text* (it cannot see
the image). It leans on the reference solution (gold CoT) to compare the model's
claimed observations against the correct ones. Treat as an indicative, LLM-assisted
qualitative pass — spot-validate the per-item labels in results_errormode.jsonl.

Usage: python babyvision_errormode.py --port 30001 --base <dir>
"""

import argparse
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DRAWS = [("results_a0_nothink", 1), ("results_standard", 1), ("results_standard", 2),
         ("results_standard", 3), ("results_a3_forced_long", 1),
         ("results_b1_reinject", 1), ("results_b2_noreinject", 1)]

PROMPT = """You are analyzing WHY a vision-language model got a visual puzzle WRONG.
You are given the question, the correct answer, a reference solution describing the
correct observations, and the model's full reasoning + final answer. You cannot see
the image — judge from the model's own text what kind of mistake it made.

Classify the PRIMARY cause of the error as exactly one word:

PERCEPTION - the model's description of the image is factually wrong: it miscounts,
  misses or invents elements, or gets positions / colors / shapes / relations wrong.
  The error originates in what it claims to see (its observations disagree with the
  reference solution).
REASONING - the model's description of the image is essentially accurate, but it
  draws an incorrect conclusion or makes a logical / procedural mistake from correct
  observations.
INDECISION - the model never commits to a clear single answer; it contradicts itself
  or hedges so no conclusion can be identified.
OTHER - none of the above.

Question:
{q}

Correct answer:
{gold}

Reference solution:
{cot}

Model reasoning and final answer:
{trace}

Answer with exactly one word: PERCEPTION, REASONING, INDECISION, or OTHER."""

N_CONCURRENT = 16
HEAD, TAIL = 2500, 2500   # keep head+tail of long traces


def load(base, d, p):
    f = base / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def trace_text(rec):
    th = (rec.get("thinking_trace") or "").strip()
    ans = (rec.get("answer_text") or "").strip()
    if len(th) > HEAD + TAIL:
        th = th[:HEAD] + "\n...[trace trimmed]...\n" + th[-TAIL:]
    return f"[thinking]\n{th}\n\n[final answer]\n{ans}".strip()


def classify(url, model, rec):
    cot = rec.get("gold_coT") or rec.get("coT") or "(not provided)"
    prompt = PROMPT.format(q=rec.get("question_sent", ""), gold=rec.get("gt_answer", ""),
                           cot=str(cot)[:2000], trace=trace_text(rec))
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "chat_template_kwargs": {"enable_thinking": False},
               "max_tokens": 8, "temperature": 0.0}
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()
    for cat in ("PERCEPTION", "REASONING", "INDECISION", "OTHER"):
        if cat in raw:
            return cat, raw
    return "OTHER", raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--base", default="/iopsstor/scratch/cscs/raghavthind/babyvision")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    base = Path(args.base)
    url = f"http://localhost:{args.port}"
    model = args.model or requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"Judge: {model}")

    draws = {f"{d}_{p}": load(base, d, p) for d, p in DRAWS}
    common = set.intersection(*[set(v) for v in draws.values()])
    pc = {t: sum(1 for v in draws.values() if v[t].get("grade") is True) / len(draws)
          for t in common}
    stable_wrong = [t for t in common if pc[t] == 0]
    std1 = draws["results_standard_1"]
    items = [std1[t] for t in stable_wrong if std1.get(t)]
    print(f"Classifying {len(items)} stable-wrong items (standard-pass-1 traces)...\n")

    lock, counter, out = threading.Lock(), [0], []
    def work(rec):
        try:
            cat, raw = classify(url, model, rec)
        except Exception as e:
            cat, raw = "ERROR", str(e)[:40]
        rec_out = {"taskId": rec["taskId"], "subtype": rec.get("subtype"),
                   "type": rec.get("type"), "ansType": rec.get("ansType"),
                   "gt_answer": rec.get("gt_answer"), "errormode": cat, "raw": raw}
        with lock:
            counter[0] += 1
            print(f"  [{counter[0]:3d}/{len(items)}] id={rec['taskId']:>5} "
                  f"{rec.get('subtype','')[:22]:22s} → {cat}", flush=True)
            out.append(rec_out)
        return rec_out

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        list(pool.map(work, items))

    (base / "analysis").mkdir(exist_ok=True)
    with open(base / "analysis" / "results_errormode.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    # ── tallies ──
    cats = defaultdict(int)
    for r in out:
        cats[r["errormode"]] += 1
    n = len(out)
    print("\n" + "=" * 64)
    print(f"ERROR MODES of {n} stable-wrong items")
    print("=" * 64)
    for c in ("PERCEPTION", "REASONING", "INDECISION", "OTHER", "ERROR"):
        if cats[c]:
            print(f"  {c:<12}{cats[c]:>4}  ({cats[c]/n*100:4.1f}%)")
    print("\n-- by subtype (PERCEPTION / REASONING / INDECISION / OTHER) --")
    bysub = defaultdict(lambda: defaultdict(int))
    for r in out:
        bysub[r["subtype"]][r["errormode"]] += 1
    for sub in sorted(bysub):
        d = bysub[sub]
        tot = sum(d.values())
        print(f"  {sub[:30]:<30} n={tot:<3} "
              f"P={d['PERCEPTION']} R={d['REASONING']} I={d['INDECISION']} O={d['OTHER']}")
    print("\nPer-item labels → analysis/results_errormode.jsonl (spot-validate!).")


if __name__ == "__main__":
    main()
